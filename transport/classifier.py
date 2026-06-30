"""Clip classifier — STATEFUL per-camera (đây là chỗ KHÁC drone: bài toán temporal).

Đọc vio:frames:gated THEO THỨ TỰ (1 process, 1 consumer), gom frame vào ring-buffer
riêng từng camera_id (cửa sổ trượt WIN_LEN). Cứ WIN_STRIDE frame mới → chạy model
clip-level 1 lần → XADD prob vào vio:results. prob cao → lưu snapshot + clip bằng chứng.

Vì xã chỉ vài camera, 1 process gánh tốt và in-order tự nhiên (consumer group 1
consumer). Muốn scale nhiều chục camera thì shard theo hash(camera_id) ra N process.

Dùng:
  python -m transport.classifier
"""
from __future__ import annotations

import signal
import statistics
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import redis

from . import config, envelope, store
from .model import (VioModel, even_indices, multispan_predict, person_crop_box,
                    quiet_ml_logs, suppress_native_logs)

_stop = False


def _handle_sig(*_):
    global _stop
    _stop = True


def decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def annotate(frame: np.ndarray, boxes: list, prob: float) -> np.ndarray:
    """Vẽ bbox người + nhãn prob lên snapshot bằng chứng."""
    img = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, f"VIOLENCE {prob:.0%}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img


class CamWindow:
    """Ring-buffer 1 camera: frame + meta cho cửa sổ trượt."""
    def __init__(self):
        # Buffer SPAN thời gian (WIN_SPAN = round(WIN_SECONDS*GRAB_FPS) frame) → lấy đều WIN_LEN=16
        # frame lúc phân loại (khớp training). Tránh lệch train/serve về thời gian (xem config).
        self.frames: deque = deque(maxlen=config.WIN_SPAN)
        self.persons: deque = deque(maxlen=config.WIN_SPAN)
        self.boxes: deque = deque(maxlen=config.WIN_SPAN)   # bbox người từng frame → crop vùng người
        self.boxes_last: list = []
        self.since_last = 0
        self.last_ts = 0.0
        self.last_seq = 0


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    quiet_ml_logs()
    with suppress_native_logs():        # nuốt log TRT khi nạp engine + warmup (1 lần)
        model = VioModel()
        model.predict([np.zeros((config.CLIP_SIZE, config.CLIP_SIZE, 3), np.uint8)] * config.WIN_LEN)
    media = store.MediaStore()
    r = redis.from_url(config.REDIS_URL)
    cams: dict[str, CamWindow] = defaultdict(CamWindow)
    name = "cls-1"
    print(f"[cls {name}] win={config.WIN_LEN} stride={config.WIN_STRIDE} "
          f"thr={config.VIO_THRESHOLD} consuming {config.GATED_STREAM} group {config.CG_CLS}")

    total, classified, last_log = 0, 0, time.time()
    while not _stop:
        try:
            resp = r.xreadgroup(
                groupname=config.CG_CLS, consumername=name,
                streams={config.GATED_STREAM: ">"},
                count=config.READ_COUNT, block=config.READ_BLOCK_MS,
            )
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            print(f"[cls {name}] redis err: {e}; backoff 0.5s")
            time.sleep(0.5)
            continue
        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                try:
                    r.xgroup_create(config.GATED_STREAM, config.CG_CLS, id="$", mkstream=True)
                except redis.exceptions.ResponseError:
                    pass
                continue
            raise

        if not resp:
            continue
        _, entries = resp[0]
        for eid, fields in entries:
            try:
                g = envelope.parse_gated(fields)
                img = decode(g["img"])
                if img is None:
                    raise ValueError("decode failed")
            except Exception:
                r.xack(config.GATED_STREAM, config.CG_CLS, eid)
                continue

            cam = cams[g["camera_id"]]
            cam.frames.append(img)
            cam.persons.append(g["num_persons"])
            cam.boxes.append(g["boxes"])
            cam.boxes_last = g["boxes"]
            cam.since_last += 1
            cam.last_ts = g["ts"]
            cam.last_seq = g["seq"]
            total += 1
            r.xack(config.GATED_STREAM, config.CG_CLS, eid)

            # ≥ WIN_LEN frame + đủ stride → MULTI-SPAN: max prob qua các span (đánh nhanh lẫn chậm/đám đông).
            # Dùng >= WIN_LEN (không đợi đầy span) → clip/luồng ngắn vẫn được chấm; buffer trượt tối đa WIN_SPAN.
            if len(cam.frames) >= config.WIN_LEN and cam.since_last >= config.WIN_STRIDE:
                cam.since_last = 0
                fr_all, bx_all = list(cam.frames), list(cam.boxes)
                h0, w0 = fr_all[-1].shape[:2]
                spans_frames = [round(s * config.GRAB_FPS) for s in config.WIN_SECONDS_LIST]
                prob, _ = multispan_predict(model, fr_all, bx_all, spans_frames, w0, h0)
                frames = [fr_all[i] for i in even_indices(len(fr_all), config.WIN_LEN)]  # cho snapshot/clip
                med_persons = int(statistics.median(cam.persons))
                classified += 1

                snapshot_key, clip_key = "", ""
                # Lưu bằng chứng khi prob đủ cao (off hot-path: encode trong thread).
                if prob >= config.SAVE_PROB:
                    mid = frames[len(frames) // 2]
                    snap = annotate(mid, cam.boxes_last, prob)
                    snapshot_key = media.save_snapshot_async(g["camera_id"], cam.last_ts, snap)
                    clip_key = media.save_clip_async(g["camera_id"], cam.last_ts, frames)

                result = {
                    "prob": round(prob, 4),
                    "num_persons": med_persons,
                    "ts": cam.last_ts,
                    "seq": cam.last_seq,
                    "snapshot_key": snapshot_key,
                    "clip_key": clip_key,
                    "model": config.VIO_BACKBONE,
                    "modelVersion": config.MODEL_VERSION,
                }
                r.xadd(config.RES_STREAM, envelope.build_res(g["camera_id"], result),
                       maxlen=config.RES_MAXLEN, approximate=True)

        now = time.time()
        if now - last_log >= 2.0:
            print(f"[cls {name}] frames={total} classified={classified} cams={len(cams)} "
                  f"xlen={r.xlen(config.GATED_STREAM)}")
            last_log = now

    media.close()
    print(f"[cls {name}] stopped. frames={total} classified={classified}")


if __name__ == "__main__":
    main()
