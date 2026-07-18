"""Dashboard — nhúng thẳng pipeline camera live (transport.live) + ráp logic ALERT
(k-of-n + cooldown) + đẩy SSE realtime. KHÔNG qua Redis (xem transport/live.py) — camera nối
trực tiếp vào tiến trình này.

Đây là "BE" của pipeline: smoothing chống false-positive + cooldown chống spam là việc của
tầng này, không phải classifier. Human-in-the-loop: dashboard chỉ SÁNG ĐÈN + hiện snapshot/clip
để trực ban xác nhận, KHÔNG tự động điều người.

Chạy:
  LIVE_CAMS="cam1=rtsp://user:pass@ip/stream" python -m dashboard.app
  # rồi mở http://<server>:8080
  # LIVE_CAMS rỗng/không set → chỉ chức năng upload, không có camera live.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime

import base64

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from transport import config, live

app = FastAPI(title="Phát hiện bạo lực — Như Quỳnh")

# --- Bảo vệ truy cập: HTTP Basic (bật khi DASH_PASS có giá trị) ---------------
DASH_USER = os.getenv("DASH_USER", "admin")
DASH_PASS = os.getenv("DASH_PASS", "")  # rỗng = mở (dev); đặt giá trị để khóa public


@app.middleware("http")
async def _basic_auth(request, call_next):
    if DASH_PASS:
        hdr = request.headers.get("authorization", "")
        ok = False
        if hdr.startswith("Basic "):
            try:
                u, p = base64.b64decode(hdr[6:]).decode("utf-8").split(":", 1)
                ok = (u == DASH_USER and p == DASH_PASS)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response("Cần đăng nhập", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Nhu Quynh"'})
    return await call_next(request)

os.makedirs(config.MEDIA_DIR, exist_ok=True)
os.makedirs(os.path.join(config.MEDIA_DIR, "analyzed"), exist_ok=True)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR), name="media")

# --- Upload & phân tích offline (tải video → infer → xem lại có detection) --
_jobs: dict[str, dict] = {}          # job_id → {status, progress, name, result, video_url, error}
_jobs_order: deque = deque(maxlen=50)
_analyze_lock = threading.Lock()      # chạy tuần tự, tránh 2 luồng nạp engine đua nhau


def _run_analysis(job_id: str, in_path: str) -> None:
    from transport import analyze
    job = _jobs[job_id]
    try:
        with _analyze_lock:
            job["status"] = "running"
            out_rel = f"analyzed/{job_id}.mp4"
            out_path = os.path.join(config.MEDIA_DIR, out_rel)

            # Đẩy tiến độ qua SSE (throttle ~0.4s) thay vì để client poll → bớt request, hết lag.
            last_bcast = [0.0]

            def prog(p):
                job["progress"] = round(p, 3)
                now = time.time()
                if p >= 1.0 or now - last_bcast[0] >= 0.4:
                    last_bcast[0] = now
                    _broadcast({"type": "progress", "id": job_id, "progress": job["progress"]})

            result = analyze.analyze_video(in_path, out_path, on_progress=prog)
        job.update(status="done", progress=1.0, result=result, video_url=f"/media/{out_rel}",
                   poster_url=f"/media/analyzed/{job_id}.jpg")
        _broadcast({"type": "analysis", "job": _job_public(job)})
        print(f"[upload] ✓ {job['name']} → violence={result['violence']} max={result['max_prob']}")
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))
        print(f"[upload] ✗ {job['name']}: {e}")


def _job_public(j: dict) -> dict:
    return {k: j.get(k) for k in ("id", "name", "status", "progress", "result",
                                  "video_url", "poster_url", "error", "time")}

# --- Trạng thái chia sẻ giữa thread camera live và web ----------------------
_state_lock = threading.Lock()
_live: dict[str, dict] = {}                 # camera_id → {prob, num_persons, ts}
_alerts: deque = deque(maxlen=100)          # alert gần nhất (mới ở đầu)
_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.ALERT_N))
_last_alert: dict[str, float] = {}
_last_positive: dict[str, dict] = {}  # camera_id → result DƯƠNG gần nhất (prob cao + có media)
_strong_streak: dict[str, int] = defaultdict(int)  # số cửa sổ MẠNH liên tiếp (đường mạnh cần streak)
_subscribers: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None
_alert_seq = 0


def _broadcast(event: dict) -> None:
    """Đẩy event tới mọi client SSE (gọi từ thread camera live → schedule lên event loop)."""
    if _loop is None:
        return
    data = json.dumps(event, ensure_ascii=False)
    for q in list(_subscribers):
        _loop.call_soon_threadsafe(q.put_nowait, data)


def _media_url(key: str) -> str:
    return f"/media/{key}" if key else ""


def _handle_live_result(camera_id: str, result: dict) -> None:
    """Callback gọi TRỰC TIẾP từ thread camera (transport.live) mỗi lần có kết quả classify mới
    — cập nhật live + áp k-of-n/cooldown → alert. Cùng process, không qua message bus."""
    global _alert_seq
    prob = result["prob"]
    now = time.time()
    enough_persons = result["num_persons"] >= config.MIN_PERSONS
    positive = prob >= config.VIO_THRESHOLD and enough_persons
    strong = prob >= config.VIO_THRESHOLD_STRONG and enough_persons

    with _state_lock:
        _live[camera_id] = {"prob": prob, "num_persons": result["num_persons"], "ts": result["ts"]}
        w = _windows[camera_id]
        w.append(1 if positive else 0)
        if positive:
            _last_positive[camera_id] = result
        # Đường mạnh cần VIO_STRONG_STREAK cửa sổ mạnh LIÊN TIẾP — spike 1 cửa sổ (lắc cam,
        # đứng/ngồi đột ngột) không nổ được; cú đánh thật vẫn cao nhiều cửa sổ liền (xem config).
        _strong_streak[camera_id] = _strong_streak[camera_id] + 1 if strong else 0
        # Alert khi: đủ K/N cửa sổ dương (kéo dài) HOẶC đủ streak cửa sổ mạnh (chớp nhoáng,
        # không để k-of-n nuốt mất). Cả hai vẫn tôn trọng cooldown chống spam.
        fire = ((sum(w) >= config.ALERT_K
                 or _strong_streak[camera_id] >= config.VIO_STRONG_STREAK)
                and now - _last_alert.get(camera_id, 0) >= config.COOLDOWN_S)
        alert = None
        if fire:
            _last_alert[camera_id] = now
            _alert_seq += 1
            # k-of-n có thể chỉ đủ K Ở CỬA SỔ HIỆN TẠI KHÔNG DƯƠNG (vd 3 dương rồi 1 cửa sổ
            # nguội mới kích vì cooldown vừa hết) → alert phải mang prob + snapshot/clip của
            # cửa sổ DƯƠNG gần nhất, không phải cửa sổ kích hoạt (từng thấy alert prob=0.02
            # không có ảnh — vô nghĩa với trực ban).
            ev = result if positive else _last_positive.get(camera_id, result)
            alert = {
                "id": _alert_seq,
                "camera_id": camera_id,
                "time": datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S %d/%m"),
                "prob": ev["prob"],
                "num_persons": ev["num_persons"],
                "snapshot_url": _media_url(ev.get("snapshot_key", "")),
                "clip_url": _media_url(ev.get("clip_key", "")),
            }
            _alerts.appendleft(alert)

    _broadcast({"type": "live", "camera_id": camera_id,
                "prob": prob, "num_persons": result["num_persons"]})
    if alert:
        print(f"[dash] 🚨 ALERT {camera_id} prob={prob:.2f} persons={result['num_persons']}")
        _broadcast({"type": "alert", "alert": alert})


def _live_cams_loop() -> None:
    """Thread: nạp model + chạy camera live (transport.live), gọi _handle_live_result mỗi kết
    quả. Không có LIVE_CAMS → bỏ qua, dashboard vẫn chạy được chức năng upload."""
    cams = live.parse_cams_env(config.LIVE_CAMS)
    if not cams:
        print("[dash] LIVE_CAMS trống — không chạy camera live (chỉ upload).")
        return
    live.run_live(cams, _handle_live_result, loop=config.LIVE_LOOP)


def _cleanup_loop() -> None:
    """Thread: dọn media quá TTL (Nghị định 13 — không giữ dữ liệu cá nhân quá hạn)."""
    from transport.store import MediaStore
    ms = MediaStore()
    while True:
        time.sleep(3600)
        n = ms.cleanup()
        if n:
            print(f"[dash] cleanup: xóa {n} file media quá hạn")


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    # Gắn filter SAU khi uvicorn cấu hình logging (dictConfig lúc khởi động) → không bị reset.
    logging.getLogger("uvicorn.access").addFilter(_QuietStaticAccess())
    threading.Thread(target=_live_cams_loop, daemon=True, name="live-cams").start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


@app.get("/api/state")
async def state():
    with _state_lock:
        return {
            "live": _live,
            "alerts": list(_alerts),
            "config": {"threshold": config.VIO_THRESHOLD,
                       "threshold_strong": config.VIO_THRESHOLD_STRONG,
                       "k": config.ALERT_K, "n": config.ALERT_N,
                       "cooldown_s": config.COOLDOWN_S},
        }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Nhận video, lưu disk, chạy phân tích nền (YOLO gate + classifier) → trả job_id."""
    job_id = uuid.uuid4().hex[:12]
    safe = os.path.basename(file.filename or "video.mp4").replace(" ", "_")
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe}")
    # Ghi file trong threadpool — KHÔNG block event loop (đang async) → SSE/UI không khựng khi upload.
    def _save():
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    await run_in_threadpool(_save)
    job = {"id": job_id, "name": safe, "status": "queued", "progress": 0.0,
           "result": None, "video_url": "", "poster_url": "", "error": "",
           "in_path": in_path, "time": datetime.now().strftime("%H:%M:%S %d/%m")}
    _jobs[job_id] = job
    _jobs_order.appendleft(job_id)
    threading.Thread(target=_run_analysis, args=(job_id, in_path), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/analyses")
async def analyses():
    return [_job_public(_jobs[i]) for i in _jobs_order if i in _jobs]


@app.delete("/api/analyses/{job_id}")
async def delete_analysis(job_id: str):
    """Xóa job + video upload nguồn + video phân tích + poster (giải phóng dung lượng)."""
    import glob
    job = _jobs.pop(job_id, None)
    try:
        _jobs_order.remove(job_id)
    except ValueError:
        pass
    removed = []
    paths = glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}_*"))         # video upload gốc
    paths += [os.path.join(config.MEDIA_DIR, "analyzed", f"{job_id}.mp4"),
              os.path.join(config.MEDIA_DIR, "analyzed", f"{job_id}.jpg")]  # output + poster
    if job and job.get("in_path"):
        paths.append(job["in_path"])
    for p in set(paths):
        try:
            if os.path.exists(p):
                os.remove(p)
                removed.append(os.path.basename(p))
        except OSError:
            pass
    return {"ok": True, "removed": removed}


@app.get("/stream/{camera_id}")
async def stream(camera_id: str):
    """Live view MJPEG: đẩy frame mới nhất của camera (~12fps, resize + JPEG) — <img> trên UI
    trỏ thẳng vào đây. Chỉ encode khi frame đổi (theo seq), encode chạy trong threadpool."""
    async def gen():
        last_seq = -1
        while True:
            data = await run_in_threadpool(live.latest_jpeg, camera_id)
            if data is not None:
                jpg, seq = data
                if seq != last_seq:
                    last_seq = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
            await asyncio.sleep(1 / 12)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/events")
async def events():
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.append(q)

    async def gen():
        try:
            yield "retry: 3000\n\n"
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        finally:
            if q in _subscribers:
                _subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#e23b3b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3a12 12 0 0 0 8.5 3A12 12 0 0 1 12 21 12 12 0 0 1 3.5 6 12 12 0 0 0 12 3"/>'
    '<path d="m9 12 2 2 4-4"/></svg>'
)


@app.get("/favicon.ico")
async def favicon():
    # Trả icon khiên (SVG) → hết 404 nhiễu log; browser hiện icon ở tab.
    return Response(_FAVICON, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML

class _QuietStaticAccess(logging.Filter):
    """Bỏ access-log của uvicorn cho ảnh/clip tĩnh + favicon (rác, lặp mỗi lần load thumbnail).

    Giữ nguyên log API/upload/SSE — chỉ lọc /media/ và /favicon.ico.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        for a in record.args or ():
            if isinstance(a, str) and (a.startswith("/media/") or a == "/favicon.ico"):
                return False
        return True


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=config.DASH_HOST, port=config.DASH_PORT)


_HTML = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phát hiện bạo lực — Như Quỳnh</title>
<link rel="icon" href="/favicon.ico" type="image/svg+xml">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.21.0/dist/tabler-icons.min.css">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --page: #eef2f6;
  --ink: #17202f;
  --muted: #68758a;
  --soft: #f8fafc;
  --panel: #ffffff;
  --line: #d8e0ea;
  --line-strong: #c5cfdd;
  --ok: #168a58;
  --warn: #b7791f;
  --bad: #d02f2f;
  --blue: #2458d3;
  --teal: #0f766e;
  --radius: 8px;
  --shadow: 0 16px 40px rgba(38, 50, 72, .10);
}
body {
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page);
  color: var(--ink);
  font-size: 14px;
}
button, input { font: inherit; }
button { border: 0; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  display: flex;
  align-items: center;
  gap: 18px;
  min-height: 68px;
  padding: 12px 22px;
  color: #fff;
  background: #162033;
  border-bottom: 1px solid rgba(255,255,255,.10);
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 260px;
}
.brand-mark {
  width: 40px;
  height: 40px;
  border-radius: var(--radius);
  display: grid;
  place-items: center;
  color: #fff;
  background: #2458d3;
  box-shadow: inset 0 -10px 18px rgba(0,0,0,.12);
  flex-shrink: 0;
}
.brand-mark i { font-size: 23px; }
.brand-title { font-weight: 800; line-height: 1.15; }
.brand-sub { color: #aab7ca; font-size: 12px; margin-top: 2px; }
.top-actions {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  min-width: 0;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 32px;
  padding: 0 10px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: var(--radius);
  color: #dce5f2;
  background: rgba(255,255,255,.07);
  font-size: 12px;
  white-space: nowrap;
}
.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #9aa7ba;
  transition: background .25s, box-shadow .25s;
  flex-shrink: 0;
}
.dot.ok { background: #29c47b; box-shadow: 0 0 0 4px rgba(41,196,123,.16); }
.dot.err { background: #ff6464; box-shadow: 0 0 0 4px rgba(255,100,100,.18); }
.shell {
  width: min(1540px, 100%);
  margin: 0 auto;
  padding: 18px;
}
.overview {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.stat {
  min-height: 92px;
  padding: 15px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.stat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.stat-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0;
}
.stat i {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: var(--radius);
  color: var(--blue);
  background: #e9efff;
  font-size: 18px;
}
.stat-value {
  margin-top: 12px;
  font-size: 28px;
  font-weight: 850;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.stat-foot {
  color: var(--muted);
  font-size: 12px;
  margin-top: 6px;
}
.stat.alert i { color: var(--bad); background: #ffecec; }
.stat.warn i { color: var(--warn); background: #fff4d8; }
.stat.teal i { color: var(--teal); background: #e3f6f3; }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 390px;
  gap: 16px;
  align-items: start;
}
.main { display: grid; gap: 16px; min-width: 0; }
.section, .alerts-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 15px 16px;
  border-bottom: 1px solid var(--line);
}
.section-title {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 14px;
  font-weight: 800;
}
.section-title i { color: var(--blue); font-size: 18px; }
.section-note { color: var(--muted); font-size: 12px; white-space: nowrap; }
.upload-zone {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}
.file-btn {
  min-height: 52px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 14px;
  background: var(--soft);
  border: 1px dashed var(--line-strong);
  border-radius: var(--radius);
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.file-btn:hover { border-color: var(--blue); background: #f4f7ff; }
.file-btn input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
  font-size: 1px;
}
.file-icon {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: var(--radius);
  background: #e3f6f3;
  color: var(--teal);
  flex-shrink: 0;
}
.file-main { min-width: 0; }
.file-name {
  color: var(--ink);
  font-weight: 750;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file-hint, .up-msg { color: var(--muted); font-size: 12px; margin-top: 2px; }
.btn {
  min-height: 52px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 0 18px;
  border-radius: var(--radius);
  background: var(--blue);
  color: #fff;
  font-weight: 800;
  cursor: pointer;
  transition: transform .15s, opacity .2s, background .2s;
  white-space: nowrap;
}
.btn:hover { background: #1d49b8; transform: translateY(-1px); }
.btn:disabled { opacity: .45; cursor: default; transform: none; }
.ana-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 12px;
  padding: 14px 16px 16px;
}
.a-card {
  position: relative;
  overflow: hidden;
  min-height: 205px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  cursor: pointer;
  transition: transform .15s, border-color .2s, box-shadow .2s;
}
.a-card:hover { border-color: #9fb2d7; box-shadow: 0 12px 28px rgba(36,88,211,.12); transform: translateY(-2px); }
.a-thumb, .a-ph {
  width: 100%;
  height: 116px;
  display: block;
  background: #151b27;
}
.a-thumb { object-fit: cover; }
.a-ph {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #8fa1b8;
}
.a-ph i { font-size: 30px; }
.a-body { padding: 10px 11px 12px; }
.a-name {
  color: var(--ink);
  font-size: 13px;
  font-weight: 760;
  line-height: 1.35;
  margin-bottom: 8px;
  word-break: break-word;
}
.a-meta { color: var(--muted); font-size: 12px; margin-top: 7px; }
.a-del {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 28px;
  height: 28px;
  border-radius: var(--radius);
  display: grid;
  place-items: center;
  background: rgba(255,255,255,.92);
  border: 1px solid rgba(23,32,47,.12);
  color: var(--muted);
  cursor: pointer;
  z-index: 2;
}
.a-del:hover { color: var(--bad); border-color: #f0b4b4; background: #fff3f3; }
.bdg {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 8px;
  border-radius: var(--radius);
  font-size: 12px;
  font-weight: 800;
}
.bdg i { font-size: 13px; }
.bdg-bad { color: var(--bad); background: #fff0f0; border: 1px solid #ffcaca; }
.bdg-ok { color: var(--ok); background: #eaf8f1; border: 1px solid #bfe8d2; }
.bdg-run { color: var(--warn); background: #fff6df; border: 1px solid #f0d48c; }
.cam-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  padding: 14px 16px 16px;
}
.c-card {
  display: grid;
  gap: 14px;
  min-height: 168px;
  padding: 14px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  transition: border-color .25s, box-shadow .25s;
}
.c-card.w { border-color: #edc76d; box-shadow: inset 0 0 0 1px #ffe7a3; }
.c-card.d { border-color: #f2a4a4; box-shadow: inset 0 0 0 1px #ffd2d2; animation: riskPulse 1.9s ease-in-out infinite; }
@keyframes riskPulse {
  0%, 100% { box-shadow: inset 0 0 0 1px #ffd2d2, 0 0 0 rgba(208,47,47,0); }
  50% { box-shadow: inset 0 0 0 1px #ffb1b1, 0 0 0 5px rgba(208,47,47,.07); }
}
.c-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.c-name {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-weight: 850;
}
.c-name i {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: var(--radius);
  color: var(--blue);
  background: #e9efff;
  flex-shrink: 0;
}
.c-name span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.c-status {
  padding: 4px 8px;
  border-radius: var(--radius);
  color: var(--ok);
  background: #eaf8f1;
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}
.c-status.warn { color: var(--warn); background: #fff6df; }
.c-status.danger { color: var(--bad); background: #fff0f0; }
.risk-row {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 12px;
}
.risk-label { color: var(--muted); font-size: 12px; font-weight: 700; }
.risk-value { font-size: 34px; line-height: .95; font-weight: 900; font-variant-numeric: tabular-nums; }
.meter {
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: #e7edf5;
}
.meter i {
  display: block;
  width: 0;
  height: 100%;
  border-radius: 999px;
  background: var(--ok);
  transition: width .35s ease, background .35s ease;
}
.c-video {
  width: 100%;
  aspect-ratio: 16 / 9;
  display: block;
  object-fit: cover;
  background: #151b27;
  border-radius: var(--radius);
}
.c-persons {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 650;
}
.alerts-panel {
  position: sticky;
  top: 84px;
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 104px);
  overflow: hidden;
}
.a-panel-hdr {
  padding: 15px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #fff;
  background: #8f1f1f;
}
.a-panel-hdr h2 {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 850;
}
.a-panel-sub { color: #ffdada; font-size: 12px; margin-top: 2px; }
.acnt {
  display: none;
  min-width: 28px;
  height: 28px;
  place-items: center;
  border-radius: var(--radius);
  color: #8f1f1f;
  background: #fff;
  font-size: 12px;
  font-weight: 900;
}
.acnt.on { display: grid; }
.a-scroll {
  overflow-y: auto;
  flex: 1;
  padding: 12px;
  display: grid;
  gap: 10px;
  align-content: start;
  background: #fff8f8;
}
.al-card {
  overflow: hidden;
  border: 1px solid #f2b6b6;
  border-radius: var(--radius);
  background: #fff;
  cursor: pointer;
  animation: fadein .24s ease;
}
@keyframes fadein { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }
.al-card:hover { border-color: #d87575; }
.al-thumb, .al-ph {
  width: 100%;
  height: 128px;
  display: block;
  background: #2b1111;
}
.al-thumb { object-fit: cover; }
.al-ph {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffb4b4;
}
.al-ph i { font-size: 32px; }
.al-info { padding: 11px 12px 12px; }
.al-prob {
  color: var(--bad);
  font-size: 18px;
  font-weight: 900;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.al-sub {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--muted);
  font-size: 12px;
  margin-top: 5px;
}
.al-chips {
  display: flex;
  gap: 6px;
  margin-top: 9px;
  flex-wrap: wrap;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 24px;
  padding: 0 8px;
  color: #4d5b70;
  background: #f4f7fb;
  border: 1px solid #e2e8f1;
  border-radius: var(--radius);
  font-size: 12px;
  font-weight: 700;
}
.chip-btn { color: var(--blue); background: #eef3ff; border-color: #c9d8ff; }
.chip-btn:hover { border-color: var(--blue); }
.empty {
  grid-column: 1 / -1;
  color: var(--muted);
  font-size: 13px;
  text-align: center;
  padding: 32px 10px;
}
.modal {
  position: fixed;
  inset: 0;
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 22px;
  background: rgba(15, 23, 42, .78);
}
.modal.open { display: flex; }
.modal-box {
  width: 100%;
  max-width: min(92vw, 940px);
  overflow: hidden;
  background: #fff;
  border-radius: var(--radius);
  box-shadow: 0 30px 80px rgba(0,0,0,.35);
}
.modal-hdr {
  min-height: 50px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
}
.modal-hdr h3 {
  color: var(--ink);
  font-size: 14px;
  font-weight: 800;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.modal-x {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: var(--radius);
  background: #f4f7fb;
  border: 1px solid var(--line);
  color: var(--muted);
  cursor: pointer;
  flex-shrink: 0;
}
.modal-x:hover { color: var(--ink); background: #eef3ff; }
.modal-body { padding: 14px; background: #f8fafc; }
.modal-body video, .modal-body img {
  width: 100%;
  display: block;
  max-height: calc(84vh - 80px);
  object-fit: contain;
  background: #000;
  border-radius: var(--radius);
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c8d2df; border-radius: 999px; }
@media (max-width: 1080px) {
  .overview { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .layout { grid-template-columns: 1fr; }
  .alerts-panel { position: static; max-height: none; }
}
@media (max-width: 720px) {
  .topbar { align-items: flex-start; flex-direction: column; gap: 12px; padding: 14px; }
  .brand { min-width: 0; }
  .top-actions { width: 100%; justify-content: flex-start; flex-wrap: wrap; }
  .shell { padding: 12px; }
  .overview { grid-template-columns: 1fr; }
  .section-head { align-items: flex-start; flex-direction: column; }
  .section-note { white-space: normal; }
  .upload-zone { grid-template-columns: 1fr; }
  .btn { width: 100%; }
  .cam-grid, .ana-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <div class="brand-mark"><i class="ti ti-shield-lock" aria-hidden="true"></i></div>
    <div>
      <div class="brand-title">Trung tâm giám sát Như Quỳnh</div>
      <div class="brand-sub">Phát hiện hành vi bạo lực theo thời gian thực</div>
    </div>
  </div>
  <div class="top-actions">
    <span class="pill"><span class="dot" id="conn"></span><span id="connText">Đang kết nối</span></span>
  </div>
</header>

<main class="shell">
  <section class="overview" aria-label="Tổng quan">
    <div class="stat teal">
      <div class="stat-row">
        <span class="stat-label">Camera online</span>
        <i class="ti ti-video" aria-hidden="true"></i>
      </div>
      <div class="stat-value" id="stat-cams">0</div>
      <div class="stat-foot">Luồng đang gửi tín hiệu</div>
    </div>
    <div class="stat warn">
      <div class="stat-row">
        <span class="stat-label">Nguy cơ cao nhất</span>
        <i class="ti ti-activity" aria-hidden="true"></i>
      </div>
      <div class="stat-value" id="stat-risk">0%</div>
      <div class="stat-foot" id="stat-risk-cam">Chưa có dữ liệu</div>
    </div>
    <div class="stat alert">
      <div class="stat-row">
        <span class="stat-label">Cảnh báo</span>
        <i class="ti ti-alert-triangle" aria-hidden="true"></i>
      </div>
      <div class="stat-value" id="stat-alerts">0</div>
      <div class="stat-foot">Sự kiện trong phiên này</div>
    </div>
    <div class="stat">
      <div class="stat-row">
        <span class="stat-label">Video offline</span>
        <i class="ti ti-file-analytics" aria-hidden="true"></i>
      </div>
      <div class="stat-value" id="stat-queue">0</div>
      <div class="stat-foot">Hồ sơ đã nạp</div>
    </div>
  </section>

  <div class="layout">
    <div class="main">
      <section class="section">
        <div class="section-head">
          <div class="section-title"><i class="ti ti-upload" aria-hidden="true"></i>Phân tích video ngoài luồng</div>
          <div class="section-note">Tải clip để xem lại kết quả nhận diện và bằng chứng</div>
        </div>
        <div class="upload-zone">
        <label class="file-btn">
          <span class="file-icon"><i class="ti ti-file-video" aria-hidden="true"></i></span>
          <span class="file-main">
            <span class="file-name" id="fname">Chọn video...</span>
            <span class="file-hint" id="upmsg">MP4 hoặc video từ camera lưu trữ</span>
          </span>
          <input type="file" id="file" accept="video/*">
        </label>
        <button class="btn" id="send"><i class="ti ti-player-play" aria-hidden="true"></i>Phân tích</button>
        </div>
        <div class="ana-grid" id="ana">
          <div class="empty">Chưa có video nào được phân tích.</div>
        </div>
      </section>

      <section class="section">
        <div class="section-head">
          <div class="section-title"><i class="ti ti-video" aria-hidden="true"></i>Camera trực tiếp</div>
          <div class="section-note">Theo dõi xác suất bạo lực theo từng luồng</div>
        </div>
        <div class="cam-grid" id="cams">
          <div class="empty">Chưa có dữ liệu camera...</div>
        </div>
      </section>
    </div>

    <aside class="alerts-panel">
      <div class="a-panel-hdr">
        <div>
          <h2><i class="ti ti-alert-triangle" aria-hidden="true"></i>Cảnh báo gần đây</h2>
          <div class="a-panel-sub">Snapshot và clip bằng chứng</div>
        </div>
        <span class="acnt" id="acnt">0</span>
      </div>
      <div class="a-scroll" id="alerts">
        <div class="empty">Chưa có cảnh báo.</div>
      </div>
    </aside>
  </div>
</main>

<div class="modal" id="modal">
  <div class="modal-box">
    <div class="modal-hdr">
      <h3 id="mtitle"></h3>
      <button class="modal-x" id="mclose" aria-label="Đóng">
        <i class="ti ti-x"></i>
      </button>
    </div>
    <div class="modal-body">
      <video id="mvid" controls autoplay style="display:none"></video>
      <img id="mimg" alt="" style="display:none">
    </div>
  </div>
</div>

<script>
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);
}

function riskColor(p) { return p >= 0.7 ? 'var(--bad)' : p >= 0.4 ? 'var(--warn)' : 'var(--ok)'; }
function riskText(p) { return p >= 0.7 ? 'Nguy cấp' : p >= 0.4 ? 'Theo dõi' : 'Ổn định'; }
function updateSummary() {
  const vals = Object.entries(camState);
  const highest = vals.reduce((best, cur) => cur[1].prob > best[1].prob ? cur : best, ['', { prob: 0 }]);
  document.getElementById('stat-cams').textContent = vals.length;
  document.getElementById('stat-risk').textContent = `${Math.round(highest[1].prob * 100)}%`;
  document.getElementById('stat-risk-cam').textContent = highest[0] ? highest[0] : 'Chưa có dữ liệu';
  document.getElementById('stat-alerts').textContent = alertCount;
  document.getElementById('stat-queue').textContent = Object.keys(jobs).length;
}

// ── Cameras ──────────────────────────────────────────────────────────────
const camEls = {};
const camState = {};
function renderCam(id, p, np) {
  camState[id] = { prob: p, num_persons: np };
  const grid = document.getElementById('cams');
  grid.querySelector('.empty')?.remove();
  let el = camEls[id];
  if (!el) {
    // Dựng khung 1 LẦN (img MJPEG /stream không được reload mỗi lần cập nhật prob),
    // các lần sau chỉ cập nhật số liệu qua querySelector.
    el = document.createElement('div');
    el.innerHTML = `
      <div class="c-top">
        <div class="c-name"><i class="ti ti-camera" aria-hidden="true"></i><span>${esc(id)}</span></div>
        <div class="c-status"></div>
      </div>
      <img class="c-video" src="/stream/${encodeURIComponent(id)}" alt="">
      <div>
        <div class="risk-row">
          <div class="risk-label">Xác suất bạo lực</div>
          <div class="risk-value"></div>
        </div>
        <div class="meter"><i></i></div>
      </div>
      <div class="c-persons"><i class="ti ti-users" aria-hidden="true"></i><span></span></div>`;
    grid.appendChild(el);
    camEls[id] = el;
    el.querySelector('.c-video').onclick = () =>
      openModal('Trực tiếp · ' + id, '/stream/' + encodeURIComponent(id), 'img');
  }
  const pct = Math.round(p * 100);
  el.className = 'c-card' + (p >= .7 ? ' d' : p >= .4 ? ' w' : '');
  const st = el.querySelector('.c-status');
  st.className = 'c-status ' + (p >= .7 ? 'danger' : p >= .4 ? 'warn' : '');
  st.textContent = riskText(p);
  const rv = el.querySelector('.risk-value');
  rv.style.color = riskColor(p);
  rv.textContent = pct + '%';
  const bar = el.querySelector('.meter i');
  bar.style.width = pct + '%';
  bar.style.background = riskColor(p);
  el.querySelector('.c-persons span').textContent = np + ' người trong khung hình';
  updateSummary();
}

// ── Modal ─────────────────────────────────────────────────────────────────
const modal = document.getElementById('modal');
const mvid  = document.getElementById('mvid');
const mimg  = document.getElementById('mimg');

function openModal(title, url, type) {
  document.getElementById('mtitle').textContent = title;
  if (type === 'video') {
    mvid.style.display = ''; mimg.style.display = 'none';
    mvid.src = url; mvid.play().catch(() => {});
  } else {
    mimg.style.display = ''; mvid.style.display = 'none';
    mimg.src = url; mvid.pause(); mvid.src = '';
  }
  modal.classList.add('open');
}
function closeModal() {
  modal.classList.remove('open'); mvid.pause(); mvid.src = ''; mimg.src = '';
}
document.getElementById('mclose').onclick = closeModal;
modal.onclick = e => { if (e.target === modal) closeModal(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ── Alerts ────────────────────────────────────────────────────────────────
let alertCount = 0, hasAlerts = false;
function renderAlert(a) {
  const body = document.getElementById('alerts');
  if (!hasAlerts) { body.innerHTML = ''; hasAlerts = true; }
  alertCount++;
  const cnt = document.getElementById('acnt');
  cnt.textContent = alertCount; cnt.classList.add('on');
  const el = document.createElement('div');
  el.className = 'al-card';
  const clipFn = `openModal(${JSON.stringify('Clip \u00b7 ' + a.camera_id)},${JSON.stringify(a.clip_url || '')},'video')`;
  el.innerHTML = `
    ${a.snapshot_url
      ? `<img class="al-thumb" src="${a.snapshot_url}" alt="">`
      : `<div class="al-ph"><i class="ti ti-shield-exclamation" aria-hidden="true"></i></div>`}
    <div class="al-info">
      <div class="al-prob">BẠO LỰC ${Math.round(a.prob * 100)}%</div>
      <div class="al-sub"><i class="ti ti-clock" aria-hidden="true"></i>${a.time}</div>
      <div class="al-chips">
        <span class="chip"><i class="ti ti-camera" aria-hidden="true"></i>${esc(a.camera_id)}</span>
        <span class="chip"><i class="ti ti-users" aria-hidden="true"></i>${a.num_persons} người</span>
        ${a.clip_url
          ? `<span class="chip chip-btn" onclick="event.stopPropagation();${clipFn}"><i class="ti ti-player-play" aria-hidden="true"></i>Xem clip</span>`
          : ''}
      </div>
    </div>`;
  el.onclick = () => {
    if (a.clip_url) openModal('Clip \u00b7 ' + a.camera_id, a.clip_url, 'video');
    else if (a.snapshot_url) openModal('\u1ea2nh \u00b7 ' + a.camera_id, a.snapshot_url, 'img');
  };
  body.prepend(el);
  updateSummary();
}

// ── Analyses ──────────────────────────────────────────────────────────────
const jobs = {};
const deleted = new Set();   // id đã xóa → chặn SSE/resync dựng lại card (hết phải reload)
function getBadge(j) {
  if (j.status === 'done') {
    const r = j.result || {};
    return r.violence
      ? `<span class="bdg bdg-bad"><i class="ti ti-alert-triangle"></i>Bạo lực ${Math.round(r.max_prob * 100)}%</span>`
      : `<span class="bdg bdg-ok"><i class="ti ti-circle-check"></i>Bình thường</span>`;
  }
  if (j.status === 'error') return `<span class="bdg bdg-bad"><i class="ti ti-x"></i>Lỗi</span>`;
  return `<span class="bdg bdg-run"><i class="ti ti-loader-2"></i>${Math.round((j.progress || 0) * 100)}%</span>`;
}
function upsertAna(j) {
  if (deleted.has(j.id)) return;          // đã xóa → bỏ qua event/resync đến muộn
  jobs[j.id] = j;
  const grid = document.getElementById('ana');
  grid.querySelector('.empty')?.remove();
  let el = document.getElementById('ana-' + j.id);
  if (!el) {
    el = document.createElement('div');
    el.className = 'a-card';
    el.id = 'ana-' + j.id;
    grid.prepend(el);
  }
  const r = j.result || {};
  el.innerHTML = `
    ${j.poster_url
      ? `<img class="a-thumb" src="${j.poster_url}" alt="">`
      : `<div class="a-ph"><i class="ti ti-video" aria-hidden="true"></i></div>`}
    <button class="a-del" title="Xóa" onclick="event.stopPropagation();delJob('${j.id}')">
      <i class="ti ti-x"></i>
    </button>
    <div class="a-body">
      <div class="a-name">${esc(j.name)}</div>
      ${getBadge(j)}
      <div class="a-meta">${j.status === 'done'
        ? `Đỉnh ${Math.round(r.max_prob * 100)}% @ ${r.max_at}s`
        : (j.time || '')}</div>
    </div>`;
  el.onclick = () => {
    if (j.status === 'done' && j.video_url) openModal(j.name, j.video_url, 'video');
    else if (j.status === 'error') alert('Lỗi: ' + j.error);
  };
  updateSummary();
}
async function delJob(id) {
  deleted.add(id);                        // chốt trước fetch: event đến muộn không dựng lại
  const el = document.getElementById('ana-' + id);
  if (el) { el.style.opacity = '.25'; el.style.pointerEvents = 'none'; }
  await fetch('/api/analyses/' + id, { method: 'DELETE' }).catch(() => {});
  el?.remove();
  delete jobs[id];
  const grid = document.getElementById('ana');
  if (!grid.querySelector('.a-card'))
    grid.innerHTML = '<div class="empty">Chưa có video nào được phân tích.</div>';
  updateSummary();
}
// Tiến độ đến qua SSE (không poll): chỉ cập nhật job đang chạy, card rẻ (chưa có poster).
function updateProg(id, p) {
  const j = jobs[id];
  if (deleted.has(id) || !j || j.status === 'done' || j.status === 'error') return;
  j.progress = p;
  upsertAna(j);
}

// ── SSE ───────────────────────────────────────────────────────────────────
const conn = document.getElementById('conn');
const es = new EventSource('/events');
es.onopen  = () => {
  conn.className = 'dot ok';
  document.getElementById('connText').textContent = 'Đang trực tuyến';
  // Reconnect: đồng bộ lại danh sách (bù event lỡ khi mất kết nối) — thay cho poll định kỳ.
  fetch('/api/analyses').then(r => r.json())
    .then(a => a.slice().reverse().forEach(j => { if (!deleted.has(j.id)) upsertAna(j); }))
    .catch(() => {});
};
es.onerror = () => {
  conn.className = 'dot err';
  document.getElementById('connText').textContent = 'Mất kết nối';
};
es.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.type === 'live')     renderCam(m.camera_id, m.prob, m.num_persons);
  else if (m.type === 'alert') {
    renderAlert(m.alert);
    try {
      const ac = new AudioContext(), osc = ac.createOscillator();
      osc.connect(ac.destination); osc.frequency.value = 880;
      osc.start(); setTimeout(() => osc.stop(), 250);
    } catch (_) {}
  }
  else if (m.type === 'progress') updateProg(m.id, m.progress);
  else if (m.type === 'analysis') upsertAna(m.job);
};

// ── Init ──────────────────────────────────────────────────────────────────
fetch('/api/state').then(r => r.json()).then(s => {
  for (const [id, v] of Object.entries(s.live)) renderCam(id, v.prob, v.num_persons);
  s.alerts.slice().reverse().forEach(renderAlert);
});
fetch('/api/analyses').then(r => r.json()).then(a => a.slice().reverse().forEach(upsertAna));

document.getElementById('file').onchange = e => {
  const f = e.target.files[0];
  document.getElementById('fname').textContent = f ? f.name : 'Chọn video...';
  if (!f) document.getElementById('upmsg').textContent = 'MP4 hoặc video từ camera lưu trữ';
};
document.getElementById('send').onclick = async () => {
  const f = document.getElementById('file').files[0];
  const msg = document.getElementById('upmsg');
  if (!f) { msg.textContent = 'Vui lòng chọn một video.'; return; }
  const fd = new FormData(); fd.append('file', f);
  document.getElementById('send').disabled = true;
  msg.textContent = 'Đang tải lên...';
  try {
    const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json();
    msg.textContent = 'Đã nhận, đang phân tích...';
    upsertAna({ id: r.job_id, name: f.name, status: 'queued', progress: 0 });
  } catch (_) { msg.textContent = 'Tải lên thất bại.'; }
  document.getElementById('send').disabled = false;
};
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
