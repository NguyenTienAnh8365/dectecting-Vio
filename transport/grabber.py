"""Frame grabber — đọc 1 nguồn video (file clip HOẶC rtsp://) → XADD vio:frames:ready.

cv2.VideoCapture nhận cả file lẫn URL RTSP y hệt nhau ⇒ demo bằng clip, production
chỉ đổi --src thành rtsp://... mà KHÔNG sửa code. Ring-buffer MAXLEN giữ RAM bounded;
gate chậm hơn camera ⇒ frame cũ rớt (đúng tinh thần realtime).

Nguồn LIVE (rtsp://, http(s)://) tự RECONNECT khi rớt kết nối (mạng CCTV chập chờn là
chuyện thường) — backoff tăng dần tới max 30s, thử lại vô hạn cho tới khi có Ctrl+C/SIGTERM.
Nguồn FILE giữ nguyên hành vi cũ (--loop tua lại đầu, không --loop thì dừng khi hết clip);
file mở lỗi thì báo lỗi ngay (retry vô ích với đường dẫn sai/file hỏng).

Dùng:
  # demo bằng clip (lặp vô hạn)
  python -m transport.grabber --src data/fight.mp4 --camera cam1 --fps 12 --loop
  # camera thật (tự reconnect khi rớt mạng)
  python -m transport.grabber --src "rtsp://user:pass@ip:554/stream" --camera cam1 --fps 12
"""
from __future__ import annotations

import argparse
import signal
import time

import cv2
import redis

from . import config, envelope

_stop = False


def _handle_sig(*_):
    global _stop
    _stop = True


def _is_live_src(src: str) -> bool:
    """URL luồng (rtsp/http) → nguồn LIVE cần tự reconnect khi rớt, khác file clip tĩnh."""
    return src.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _open_capture(src: str, is_live: bool, max_backoff: float = 30.0):
    """Mở VideoCapture. Nguồn LIVE: retry backoff tăng dần (1s→30s) tới khi thành công hoặc
    nhận tín hiệu dừng — mạng CCTV chập chờn/camera chưa sẵn sàng lúc khởi động không làm
    grabber chết hẳn. Nguồn FILE: thử 1 lần, lỗi thì trả None luôn (retry vô ích nếu sai
    đường dẫn/file hỏng)."""
    delay = 1.0
    while True:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        cap.release()
        if not is_live or _stop:
            return None
        print(f"[grabber] không mở được nguồn (thử lại sau {delay:.0f}s): {src}")
        slept = 0.0
        while slept < delay and not _stop:  # ngủ từng nhịp ngắn để Ctrl+C phản hồi nhanh
            time.sleep(min(1.0, delay - slept))
            slept += 1.0
        if _stop:
            return None
        delay = min(delay * 2, max_backoff)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="đường dẫn file clip hoặc URL rtsp://")
    ap.add_argument("--camera", default="cam1", help="camera_id")
    ap.add_argument("--fps", type=float, default=12.0, help="fps lấy mẫu đẩy vào pipeline")
    ap.add_argument("--loop", action="store_true", help="hết clip thì tua lại (demo)")
    ap.add_argument("--count", type=int, default=0, help="số frame gửi (0 = vô hạn)")
    ap.add_argument("--jpeg-q", type=int, default=80)
    args = ap.parse_args()

    is_live = _is_live_src(args.src)
    r = redis.from_url(config.REDIS_URL)
    cap = _open_capture(args.src, is_live)
    if cap is None:
        if _stop:
            return
        raise SystemExit(f"[grabber {args.camera}] không mở được nguồn: {args.src}")

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    print(f"[grabber {args.camera}] src={args.src} fps={args.fps} loop={args.loop} "
          f"live={is_live} → {config.REQ_STREAM} (maxlen~{config.REQ_MAXLEN})")

    seq = 0
    sent = 0
    t_next = time.time()
    while not _stop and (args.count == 0 or sent < args.count):
        ok, frame = cap.read()
        if not ok:
            if is_live:
                # Rớt kết nối (mạng CCTV chập chờn) → mở lại, KHÔNG dừng hẳn grabber.
                print(f"[grabber {args.camera}] mất kết nối, đang reconnect: {args.src}")
                cap.release()
                cap = _open_capture(args.src, is_live)
                if cap is None:  # chỉ None khi nhận tín hiệu dừng giữa lúc reconnect
                    break
                continue
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
