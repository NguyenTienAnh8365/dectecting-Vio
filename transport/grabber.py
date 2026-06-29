"""Frame grabber — đọc 1 nguồn video (file clip HOẶC rtsp://) → XADD vio:frames:ready.

cv2.VideoCapture nhận cả file lẫn URL RTSP y hệt nhau ⇒ demo bằng clip, production
chỉ đổi --src thành rtsp://... mà KHÔNG sửa code. Ring-buffer MAXLEN giữ RAM bounded;
gate chậm hơn camera ⇒ frame cũ rớt (đúng tinh thần realtime).

Dùng:
  # demo bằng clip (lặp vô hạn)
  python -m transport.grabber --src data/fight.mp4 --camera cam1 --fps 12 --loop
  # camera thật
  python -m transport.grabber --src "rtsp://user:pass@ip:554/stream" --camera cam1 --fps 12
"""
from __future__ import annotations

import argparse
import time

import cv2
import redis

from . import config, envelope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="đường dẫn file clip hoặc URL rtsp://")
    ap.add_argument("--camera", default="cam1", help="camera_id")
    ap.add_argument("--fps", type=float, default=12.0, help="fps lấy mẫu đẩy vào pipeline")
    ap.add_argument("--loop", action="store_true", help="hết clip thì tua lại (demo)")
    ap.add_argument("--count", type=int, default=0, help="số frame gửi (0 = vô hạn)")
    ap.add_argument("--jpeg-q", type=int, default=80)
    args = ap.parse_args()

    r = redis.from_url(config.REDIS_URL)
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"[grabber {args.camera}] không mở được nguồn: {args.src}")

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    print(f"[grabber {args.camera}] src={args.src} fps={args.fps} loop={args.loop} "
          f"→ {config.REQ_STREAM} (maxlen~{config.REQ_MAXLEN})")

    seq = 0
    sent = 0
    t_next = time.time()
    while args.count == 0 or sent < args.count:
        ok, frame = cap.read()
        if not ok:
            if args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # tua về đầu (chỉ với file)
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
        if not ok:
            continue
        ts = time.time()
        frame_id = f"{args.camera}-{seq:08d}"
        fields = envelope.build_req(frame_id, args.camera, ts, seq, buf.tobytes())
        r.xadd(config.REQ_STREAM, fields, maxlen=config.REQ_MAXLEN, approximate=True)
        seq += 1
        sent += 1
        if sent % 100 == 0:
            print(f"[grabber {args.camera}] sent={sent} xlen={r.xlen(config.REQ_STREAM)}")

        # Nhịp fps ổn định (camera thật tự nhịp; với file ta tự throttle).
        if period:
            t_next += period
            dt = t_next - time.time()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.time()  # tụt nhịp → reset, không dồn

    cap.release()
    print(f"[grabber {args.camera}] done: sent={sent}")


if __name__ == "__main__":
    main()
