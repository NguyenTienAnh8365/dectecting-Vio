"""Pipeline live IN-PROCESS — camera → gate → classifier CÙNG 1 tiến trình, KHÔNG qua Redis.

Redis 3 chặng cũ (grabber→gate→classifier qua vio:frames:ready/gated, xem grabber.py/gate.py/
classifier.py) là kế thừa từ dự án viễn thám: nhiều worker STATELESS chia nhau 1 hàng đợi ảnh
rời rạc, thứ tự không quan trọng. Dự án này camera IP kết nối TRỰC TIẾP về 1 server AI (vài
camera, không phải fleet lớn) — đi qua Redis chỉ tổ thêm JPEG encode/decode + round-trip mạng
mỗi frame, VÀ khiến backlog TÍCH LŨY (MAXLEN giữ nguyên ở đáy) thay vì tự bỏ frame cũ khi xử
lý không kịp camera (đo thực tế 2026-07-01: 30fps, 1 gate+1 classifier worker → độ trễ tăng dần
15s→26s trong 35s, không bao giờ bắt kịp).

Ở đây: mỗi camera 1 thread capture giữ "ô nhớ" (FrameBox) đúng 1 frame MỚI NHẤT — không có
hàng đợi, frame cũ chưa kịp xử lý bị ghi đè thẳng (đúng tinh thần realtime: luôn ưu tiên hiện
tại hơn lịch sử). 1 vòng lặp infer round-robin qua các camera, lấy frame mới nhất mỗi cam →
gate (YOLO người, đúng 1 ảnh/lần — tự nhiên khớp batch tĩnh=1 của engine, không cần
GATE_BATCH_SIZE chunking) → nạp cửa sổ trượt per-camera (CamWindow, tái dùng classifier.py) →
đủ WIN_LEN/WIN_STRIDE thì classify (multispan) → XADD thẳng vio:results. Stream/dashboard
KHÔNG đổi gì (envelope.build_res giữ nguyên định dạng) — chỉ chặng ingest+gate+classify bỏ Redis.

Giới hạn đã biết: spans_frames (multispan_predict) suy ra SỐ FRAME từ WIN_SECONDS_LIST*GRAB_FPS
(giả định buffer lấp đầy đúng nhịp GRAB_FPS). Nếu 1 server quá tải nhiều camera tới mức infer
chậm hơn camera gửi, frame bị bỏ (mailbox ghi đè) khiến cửa sổ trải dài hơn WIN_SECONDS_LIST
thực tế — chấp nhận được ở quy mô vài camera của dự án (gate+classify đã đo nhanh hơn nhiều so
với budget WIN_STRIDE/GRAB_FPS), nhưng cần lưu ý nếu mở rộng số camera/1 server sau này.

Dùng:
  python -m transport.live --cam cam1=data/fight.mp4 --loop
  python -m transport.live --cam cam1=rtsp://user:pass@ip/stream --cam cam2=rtsp://...
"""
from __future__ import annotations

import argparse
import signal
import statistics
import threading
import time

import cv2
import numpy as np
import redis
import torch

from . import config, envelope, grabber, store
from .classifier import CamWindow, annotate
from .gate import persons_from
from .model import VioModel, even_indices, multispan_predict, quiet_ml_logs, suppress_native_logs

_stop = False


def _handle_sig(*_):
    global _stop
    _stop = True
    grabber._stop = True  # _open_capture (tái dùng từ grabber.py) đọc cờ dừng của module đó


class FrameBox:
    """Ô nhớ 1 frame MỚI NHẤT của 1 camera — thread capture ghi, thread infer đọc.

    KHÔNG phải hàng đợi: ghi đè liên tục; seq tăng dần để bên đọc biết có frame mới hay chưa.
    Đây là điểm khác biệt cốt lõi so với Redis stream MAXLEN — backlog không thể tích lũy vì
    luôn chỉ giữ 1 frame (mới nhất thắng, cũ chưa xử lý kịp bị bỏ thẳng).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = -1

    def put(self, frame: np.ndarray, ts: float, seq: int) -> None:
        with self._lock:
            self._frame, self._ts, self._seq = frame, ts, seq

    def get(self):
        with self._lock:
            return self._frame, self._ts, self._seq


class LiveCam:
    def __init__(self, camera_id: str, src: str):
        self.id = camera_id
        self.src = src
        self.box = FrameBox()
        self.window = CamWindow()
        self.last_taken = -1  # seq cuối đã đưa qua gate (khác window.last_seq — dùng cho kết quả)


def _capture_worker(camera_id: str, src: str, fps: float, loop: bool, box: FrameBox) -> None:
    """Đọc 1 nguồn liên tục, chỉ ghi vào FrameBox (không encode/không Redis).

    Tái dùng nguyên bộ reconnect-với-backoff của grabber.py (_is_live_src/_open_capture) —
    đã test riêng cho rớt mạng RTSP. Nguồn FILE vẫn throttle theo --fps (mô phỏng thời gian
    thực khi demo bằng clip); nguồn LIVE để camera tự nhịp, không throttle.
    """
    is_live = grabber._is_live_src(src)
    cap = grabber._open_capture(src, is_live)
    if cap is None:
        if not _stop:
            print(f"[live {camera_id}] không mở được nguồn: {src}")
        return
    period = 1.0 / fps if (fps > 0 and not is_live) else 0.0
    print(f"[live {camera_id}] src={src} live={is_live} "
          f"{'fps=' + str(fps) if period else '(tự nhịp theo camera)'}")

    seq = 0
    t_next = time.time()
    while not _stop:
        ok, frame = cap.read()
        if not ok:
            if is_live:
                print(f"[live {camera_id}] mất kết nối, đang reconnect: {src}")
                cap.release()
                cap = grabber._open_capture(src, is_live)
                if cap is None:  # chỉ None khi nhận tín hiệu dừng giữa lúc reconnect
                    break
                continue
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break
        box.put(frame, time.time(), seq)
        seq += 1
        if period:
            t_next += period
            dt = t_next - time.time()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.time()  # tụt nhịp → reset, không dồn

    cap.release()
    print(f"[live {camera_id}] capture dừng. frames={seq}")


def _process_one(cam: LiveCam, gate_model, model, r, media) -> bool:
    """Lấy frame mới nhất của 1 camera (nếu có) → gate → nạp cửa sổ → classify khi đủ stride.

    Trả False nếu camera chưa có frame mới (để vòng ngoài biết lúc nào nên ngủ, tránh busy-loop).
    """
    frame, ts, seq = cam.box.get()
    if frame is None or seq == cam.last_taken:
        return False
    cam.last_taken = seq

    results = gate_model.predict(
        frame, device=config.GATE_DEVICE, imgsz=config.GATE_IMGSZ, conf=config.GATE_CONF,
        classes=[config.PERSON_CLASS], quantize=config.GATE_QUANTIZE, verbose=False,
    )
    boxes = persons_from(results[0])

    w = cam.window
    w.frames.append(frame)
    w.persons.append(len(boxes))
    w.boxes.append(boxes)
    w.boxes_last = boxes
    w.since_last += 1
    w.last_ts = ts
    w.last_seq = seq

    if len(w.frames) >= config.WIN_LEN and w.since_last >= config.WIN_STRIDE:
        w.since_last = 0
        fr_all, bx_all = list(w.frames), list(w.boxes)
        h0, w0 = fr_all[-1].shape[:2]
        spans_frames = [round(s * config.GRAB_FPS) for s in config.WIN_SECONDS_LIST]
        prob, _ = multispan_predict(model, fr_all, bx_all, spans_frames, w0, h0)
        frames = [fr_all[i] for i in even_indices(len(fr_all), config.WIN_LEN)]
        med_persons = int(statistics.median(w.persons))

        snapshot_key, clip_key = "", ""
        if prob >= config.SAVE_PROB:
            mid = frames[len(frames) // 2]
            snap = annotate(mid, w.boxes_last, prob)
            snapshot_key = media.save_snapshot_async(cam.id, ts, snap)
            clip_key = media.save_clip_async(cam.id, ts, frames)

        result = {
            "prob": round(prob, 4),
            "num_persons": med_persons,
            "ts": ts,
            "seq": seq,
            "snapshot_key": snapshot_key,
            "clip_key": clip_key,
            "model": config.VIO_BACKBONE,
            "modelVersion": config.MODEL_VERSION,
        }
        r.xadd(config.RES_STREAM, envelope.build_res(cam.id, result),
               maxlen=config.RES_MAXLEN, approximate=True)
    return True


def _run_infer_loop(cams: list[LiveCam], gate_model, model, r, media) -> None:
    total, last_log, last_empty_cache = 0, time.time(), time.time()
    while not _stop:
        did_work = False
        for cam in cams:
            if _process_one(cam, gate_model, model, r, media):
                did_work = True
                total += 1

        now = time.time()
        if now - last_log >= 2.0:
            print(f"[live] processed={total} cams={len(cams)}")
            last_log = now
        if now - last_empty_cache >= config.EMPTY_CACHE_INTERVAL_S:
            torch.cuda.empty_cache()  # xem config.EMPTY_CACHE_INTERVAL_S — dọn mốc VRAM cao nhất
            last_empty_cache = now
        if not did_work:
            time.sleep(0.005)  # không camera nào có frame mới → nghỉ ngắn, tránh busy-loop 100% CPU

    print(f"[live] infer dừng. processed={total}")


def _parse_cam_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--cam sai định dạng (cần NAME=SRC): {spec}")
    name, src = spec.split("=", 1)
    return name.strip(), src.strip()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", action="append", required=True, metavar="NAME=SRC",
                    help="camera_id=nguồn (file clip hoặc rtsp://...); lặp lại cho nhiều camera")
    ap.add_argument("--fps", type=float, default=float(config.GRAB_FPS),
                    help="fps đọc nguồn FILE (nguồn rtsp/http tự nhịp theo camera, không throttle)")
    ap.add_argument("--loop", action="store_true", help="nguồn FILE hết clip thì tua lại (demo)")
    args = ap.parse_args()

    quiet_ml_logs()
    with suppress_native_logs():        # nuốt log TRT/ultralytics lúc nạp engine + warmup (1 lần)
        from ultralytics import YOLO
        gate_model = YOLO(config.GATE_MODEL, task="detect", verbose=False)
        gate_model.predict(np.zeros((config.GATE_IMGSZ, config.GATE_IMGSZ, 3), np.uint8),
                            device=config.GATE_DEVICE, quantize=config.GATE_QUANTIZE,
                            classes=[config.PERSON_CLASS], verbose=False)
        model = VioModel()
        model.predict([np.zeros((config.CLIP_SIZE, config.CLIP_SIZE, 3), np.uint8)] * config.WIN_LEN)

    media = store.MediaStore()
    r = redis.from_url(config.REDIS_URL)
    cams = [LiveCam(*_parse_cam_arg(spec)) for spec in args.cam]

    threads = []
    for cam in cams:
        t = threading.Thread(target=_capture_worker,
                              args=(cam.id, cam.src, args.fps, args.loop, cam.box),
                              daemon=True, name=f"capture-{cam.id}")
        t.start()
        threads.append(t)

    print(f"[live] {len(cams)} camera, gate={config.GATE_MODEL} model={config.VIO_BACKBONE} "
          f"win={config.WIN_LEN} stride={config.WIN_STRIDE} → {config.RES_STREAM} "
          f"(ingest+gate+classify KHÔNG qua Redis)")
    try:
        _run_infer_loop(cams, gate_model, model, r, media)
    finally:
        media.close()
        for t in threads:
            t.join(timeout=2.0)


if __name__ == "__main__":
    main()
