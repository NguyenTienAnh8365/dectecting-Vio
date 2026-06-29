"""Person-gate — fork worker.py của Remote Sensing (stateless, consumer group, GPU).

XREADGROUP vio:frames:ready → YOLO đếm người + bbox → XADD vio:frames:gated → XACK.
Giữ nguyên pattern độ tin cậy: XPENDING+XCLAIM nhặt frame của worker chết, dead-letter
job độc, tự dựng lại group khi Redis flush.

Gate KHÔNG bỏ frame (giữ cửa sổ liên tục cho classifier) — nó GẮN số người + bbox vào
mỗi frame; việc "≥N người mới alert" do classifier/dashboard quyết. Nhiều worker chia
chung 1 GPU qua consumer group để giấu độ trễ decode/IO.

Dùng:
  python -m transport.gate
  WORKER_NAME=gate2 GATE_DEVICE=1 python -m transport.gate
"""
from __future__ import annotations

import os
import signal
import socket
import time

import cv2
import numpy as np
import redis
from ultralytics import YOLO

from . import config, envelope

_stop = False


def _handle_sig(*_):
    global _stop
    _stop = True


def consumer_name() -> str:
    return os.getenv("WORKER_NAME", f"gate-{socket.gethostname()}-{os.getpid()}")


def decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def persons_from(result) -> list[list[int]]:
    """Trả list bbox [x, y, w, h] (int) của các 'person'."""
    out = []
    boxes = result.boxes
    if boxes is None:
        return out
    xywh = boxes.xywh.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for (x, y, w, h), c in zip(xywh, cls):
        if int(c) == config.PERSON_CLASS:
            out.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
    return out


def reclaim_pending(r, name) -> list:
    """Nhặt lại entry pending idle quá lâu (worker chết) — XPENDING + XCLAIM (hợp Redis cũ)."""
    pend = r.xpending_range(config.REQ_STREAM, config.CG_GATE, min="-", max="+", count=config.READ_COUNT)
    stale = [p["message_id"] for p in pend if p["time_since_delivered"] >= config.CLAIM_MIN_IDLE_MS]
    if not stale:
        return []
    return r.xclaim(config.REQ_STREAM, config.CG_GATE, name, config.CLAIM_MIN_IDLE_MS, stale)


def drain_poison(r, claimed) -> list:
    """Job giao > MAX_DELIVERIES lần (độc/treo) → dead-letter + ack, thôi retry."""
    if not claimed:
        return claimed
    delivered = {
        p["message_id"]: p["times_delivered"]
        for p in r.xpending_range(config.REQ_STREAM, config.CG_GATE, min="-", max="+", count=len(claimed) * 2)
    }
    keep = []
    for eid, fields in claimed:
        if delivered.get(eid, 1) > config.MAX_DELIVERIES:
            r.xadd(config.DLQ_STREAM, {
                b"frame_id": fields.get(b"frame_id", b"?"),
                b"reason": f"gate exceeded {config.MAX_DELIVERIES} deliveries".encode(),
            }, maxlen=10000, approximate=True)
            r.xack(config.REQ_STREAM, config.CG_GATE, eid)
        else:
            keep.append((eid, fields))
    return keep


def process_batch(model, r, entries) -> int:
    imgs, reqs, ids, raws = [], [], [], []
    for eid, fields in entries:
        try:
            req = envelope.parse_req(fields)
            img = decode(req["img"])
            if img is None:
                raise ValueError("decode failed")
        except Exception:  # frame hỏng → ack, không kẹt pending
            r.xack(config.REQ_STREAM, config.CG_GATE, eid)
            continue
        imgs.append(img)
        reqs.append(req)
        ids.append(eid)
        raws.append(req["img"])

    if not imgs:
        return 0

    results = model.predict(
        imgs, device=config.GATE_DEVICE, imgsz=config.GATE_IMGSZ, conf=config.GATE_CONF,
        classes=[config.PERSON_CLASS], half=config.GATE_HALF, verbose=False,
    )

    for eid, req, res, raw in zip(ids, reqs, results, raws):
        boxes = persons_from(res)
        # Forward NGUYÊN bytes JPEG gốc (không re-encode) + số người + bbox cho classifier.
        r.xadd(
            config.GATED_STREAM,
            envelope.build_gated(req, len(boxes), boxes, raw),
            maxlen=config.GATED_MAXLEN, approximate=True,
        )
        r.xack(config.REQ_STREAM, config.CG_GATE, eid)
    return len(ids)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    name = consumer_name()
    print(f"[gate {name}] loading {config.GATE_MODEL} device={config.GATE_DEVICE} imgsz={config.GATE_IMGSZ}")
    model = YOLO(config.GATE_MODEL)
    model.predict(np.zeros((config.GATE_IMGSZ, config.GATE_IMGSZ, 3), np.uint8),
                  device=config.GATE_DEVICE, half=config.GATE_HALF, classes=[config.PERSON_CLASS], verbose=False)
    r = redis.from_url(config.REDIS_URL)
    print(f"[gate {name}] ready. consuming {config.REQ_STREAM} group {config.CG_GATE}")

    total, last_log = 0, time.time()
    while not _stop:
        try:
            claimed = drain_poison(r, reclaim_pending(r, name))
            if claimed:
                total += process_batch(model, r, claimed)
        except redis.exceptions.ResponseError:
            pass

        try:
            resp = r.xreadgroup(
                groupname=config.CG_GATE, consumername=name,
                streams={config.REQ_STREAM: ">"},
                count=config.READ_COUNT, block=config.READ_BLOCK_MS,
            )
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            print(f"[gate {name}] redis err: {e}; backoff 0.5s")
            time.sleep(0.5)
            continue
        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                try:
                    r.xgroup_create(config.REQ_STREAM, config.CG_GATE, id="$", mkstream=True)
                except redis.exceptions.ResponseError:
                    pass
                continue
            raise

        if resp:
            _, entries = resp[0]
            total += process_batch(model, r, entries)

        now = time.time()
        if now - last_log >= 2.0:
            print(f"[gate {name}] processed={total} xlen={r.xlen(config.REQ_STREAM)}")
            last_log = now

    print(f"[gate {name}] stopped. total={total}")


if __name__ == "__main__":
    main()
