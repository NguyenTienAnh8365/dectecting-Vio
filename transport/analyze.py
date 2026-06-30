"""Phân tích offline 1 video upload → xuất video H.264 có overlay detection + verdict.

Khác pipeline live (Redis streaming): đây là đường "tải video lên → infer → xem lại có
detection". Tái dùng đúng 2 model engine: YOLO gate (đếm/định vị người) + VioModel
(clip-level, TensorRT FP16). Xuất MP4 H.264 (imageio-ffmpeg) để trình duyệt phát được —
opencv mp4v KHÔNG phát trên web.
"""
from __future__ import annotations

import os
from collections import deque

import cv2
import imageio
import numpy as np

from . import config
from .model import (VioModel, even_indices, multispan_predict, person_crop_box,
                    quiet_ml_logs, suppress_native_logs)


def _cv2_can_read(path: str) -> bool:
    """cv2 (ffmpeg nội bộ OpenCV) đọc được frame đầu không? AV1 thường FAIL ở đây."""
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _transcode_h264(src: str, dst: str) -> None:
    """Chuyển video sang H.264 bằng ffmpeg của imageio (CÓ giải mã AV1: libdav1d) → cv2 đọc được."""
    import subprocess
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([exe, "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-an", "-loglevel", "error", dst],
                       capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("Không giải mã/chuyển mã được video (định dạng không hỗ trợ hoặc file hỏng).")


_gate = None
_model = None


def _load():
    global _gate, _model
    quiet_ml_logs()
    with suppress_native_logs():        # nuốt log TRT [I]/[W] khi nạp engine + warmup (1 lần)
        if _model is None:
            _model = VioModel()
        if _gate is None:
            from ultralytics import YOLO
            _gate = YOLO(config.GATE_MODEL, task="detect", verbose=False)
        # Warmup: ép lazy-init TRT (tạo execution context) xảy ra TRONG vùng bị nuốt log,
        # nếu không nó phát log ở lần infer đầu trong vòng lặp (ngoài suppressor).
        dummy = np.zeros((config.GATE_IMGSZ, config.GATE_IMGSZ, 3), np.uint8)
        _gate.predict(dummy, imgsz=config.GATE_IMGSZ, classes=[config.PERSON_CLASS],
                      quantize=config.GATE_QUANTIZE, device=config.GATE_DEVICE, verbose=False)
        _model.predict([dummy] * config.WIN_LEN)
    return _gate, _model


def _persons(res) -> list[tuple[int, int, int, int]]:
    out = []
    b = res.boxes
    if b is None:
        return out
    for (x, y, w, h), c in zip(b.xywh.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
        if int(c) == config.PERSON_CLASS:
            out.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
    return out


def _draw(frame, boxes, prob, alert):
    for (x, y, w, h) in boxes:
        col = (0, 0, 255) if alert else (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
    col = (0, 0, 255) if prob >= config.VIO_THRESHOLD else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"BAO LUC: {prob*100:4.0f}%  |  nguoi: {len(boxes)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if alert:
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 2, frame.shape[0] - 2), (0, 0, 255), 6)
        cv2.putText(frame, "!! CANH BAO BAO LUC", (10, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def analyze_video(in_path: str, out_path: str, on_progress=None) -> dict:
    gate, model = _load()
    # cv2 không decode được (vd codec AV1) → transcode H.264 bằng imageio-ffmpeg rồi mới đọc.
    src, tmp = in_path, None
    if not _cv2_can_read(in_path):
        tmp = os.path.splitext(in_path)[0] + "_h264.mp4"
        _transcode_h264(in_path, tmp)
        src = tmp
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 854
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    W -= W % 2; H -= H % 2  # H.264 yuv420p cần kích thước chẵn

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p",
                                ffmpeg_log_level="error")

    # Buffer SPAN thời gian (round(WIN_SECONDS*fps) frame) → lấy đều WIN_LEN=16 frame (khớp training).
    span = max(config.WIN_LEN, round(config.WIN_SECONDS * fps))
    win: deque = deque(maxlen=span)
    box_hist: deque = deque(maxlen=span)   # bbox người từng frame → crop vùng người
    since, prob, pos_windows, strong_hits, max_prob, max_t = 0, 0.0, 0, 0, 0.0, 0.0
    timeline, idx = [], 0
    poster, first_frame = None, None
    boxes: list = []                       # tái dùng giữa các lần chạy gate (overlay không nháy)
    gate_every = max(1, config.ANALYZE_GATE_EVERY)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))
        if first_frame is None:
            first_frame = frame.copy()
        win.append(frame.copy())
        since += 1
        # Phân loại khi có ≥ WIN_LEN frame (multispan tự cắt span theo số frame có) → clip NGẮN
        # (< span buffer) vẫn được chấm, không bị bỏ sót. Buffer vẫn trượt tối đa `span` frame.
        deciding = len(win) >= config.WIN_LEN and since >= config.WIN_STRIDE
        # Chạy YOLO mỗi gate_every frame, NHƯNG ép chạy đúng frame quyết định → đếm người chuẩn.
        if idx % gate_every == 0 or deciding:
            res = gate.predict(frame, imgsz=config.GATE_IMGSZ, conf=config.GATE_CONF,
                               classes=[config.PERSON_CLASS], quantize=config.GATE_QUANTIZE,
                               device=config.GATE_DEVICE, verbose=False)[0]
            boxes = _persons(res)
        box_hist.append(boxes)
        if deciding:
            since = 0
            # MULTI-SPAN: lấy MAX prob qua các span (3s & 7s…) → bắt cả đánh nhanh lẫn chậm/đám đông.
            spans_frames = [round(s * fps) for s in config.WIN_SECONDS_LIST]
            prob, _ = multispan_predict(model, list(win), list(box_hist), spans_frames, W, H)
            t = idx / fps
            enough_persons = len(boxes) >= config.MIN_PERSONS
            is_strong = prob >= config.VIO_THRESHOLD_STRONG and enough_persons
            timeline.append({"t": round(t, 2), "prob": round(prob, 3), "strong": is_strong})
            if prob >= config.VIO_THRESHOLD and enough_persons:
                pos_windows += 1
            if is_strong:                             # 1 cửa sổ mạnh đã đủ → không bỏ sót hành động ngắn
                strong_hits += 1
            if prob > max_prob:                       # giữ khung "căng" nhất làm poster
                max_prob, max_t = prob, t
                poster = (frame.copy(), list(boxes), prob)

        alert = prob >= config.VIO_THRESHOLD and len(boxes) >= config.MIN_PERSONS
        writer.append_data(cv2.cvtColor(_draw(frame, boxes, prob, alert), cv2.COLOR_BGR2RGB))
        idx += 1
        if on_progress and total and idx % 10 == 0:
            on_progress(idx / total)

    cap.release()
    writer.close()
    if tmp and os.path.exists(tmp):          # dọn file transcode tạm
        try:
            os.remove(tmp)
        except OSError:
            pass
    # Không đọc được frame nào → BÁO LỖI (không trả "Bình thường" giả: codec lạ / file hỏng).
    if idx == 0:
        raise RuntimeError("Không đọc được frame nào từ video (codec không hỗ trợ hoặc file hỏng).")
    # Poster JPG (thumbnail cho dashboard): khung prob cao nhất, hoặc khung đầu nếu không có.
    poster_path = os.path.splitext(out_path)[0] + ".jpg"
    if poster is not None:
        pf, pb, pp = poster
        cv2.imwrite(poster_path, _draw(pf, pb, pp, pp >= config.VIO_THRESHOLD), [cv2.IMWRITE_JPEG_QUALITY, 80])
    elif first_frame is not None:
        cv2.imwrite(poster_path, first_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if on_progress:
        on_progress(1.0)
    return {
        "frames": idx,
        "fps": round(fps, 1),
        # verdict = đủ K/N ở mức thấp (kéo dài) HOẶC có ≥1 cửa sổ ở mức mạnh (chớp nhoáng).
        "violence": pos_windows >= config.ALERT_K or strong_hits > 0,
        "max_prob": round(max_prob, 3),
        "max_at": round(max_t, 2),
        "pos_windows": pos_windows,
        "strong_hits": strong_hits,
        "timeline": timeline,
    }
