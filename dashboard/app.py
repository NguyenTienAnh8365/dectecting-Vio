"""Dashboard — tiêu thụ vio:results, ráp logic ALERT (k-of-n + cooldown), đẩy SSE realtime.

Đây là "BE" của pipeline (fork consumer.py): smoothing chống false-positive + cooldown
chống spam là việc của tầng này, không phải classifier. Human-in-the-loop: dashboard chỉ
SÁNG ĐÈN + hiện snapshot/clip để trực ban xác nhận, KHÔNG tự động điều người.

Chạy:
  python -m dashboard.app
  # rồi mở http://<server>:8080
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import redis
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transport import config, envelope

app = FastAPI(title="Phát hiện bạo lực — Như Quỳnh")

os.makedirs(config.MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=config.MEDIA_DIR), name="media")

# --- Trạng thái chia sẻ giữa thread Redis và web ---------------------------
_state_lock = threading.Lock()
_live: dict[str, dict] = {}                 # camera_id → {prob, num_persons, ts}
_alerts: deque = deque(maxlen=100)          # alert gần nhất (mới ở đầu)
_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.ALERT_N))
_last_alert: dict[str, float] = {}
_subscribers: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None
_alert_seq = 0


def _broadcast(event: dict) -> None:
    """Đẩy event tới mọi client SSE (gọi từ thread Redis → schedule lên event loop)."""
    if _loop is None:
        return
    data = json.dumps(event, ensure_ascii=False)
    for q in list(_subscribers):
        _loop.call_soon_threadsafe(q.put_nowait, data)


def _media_url(key: str) -> str:
    return f"/media/{key}" if key else ""


def _consume_loop() -> None:
    """Thread: XREADGROUP vio:results → cập nhật live + áp k-of-n/cooldown → alert."""
    global _alert_seq
    r = redis.from_url(config.REDIS_URL)
    name = "dash-1"
    print(f"[dash] consuming {config.RES_STREAM} group {config.CG_DASH}")
    while True:
        try:
            resp = r.xreadgroup(
                groupname=config.CG_DASH, consumername=name,
                streams={config.RES_STREAM: ">"}, count=config.READ_COUNT, block=500,
            )
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            time.sleep(0.5)
            continue
        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                try:
                    r.xgroup_create(config.RES_STREAM, config.CG_DASH, id="$", mkstream=True)
                except redis.exceptions.ResponseError:
                    pass
                continue
            raise
        if not resp:
            continue

        _, entries = resp[0]
        for eid, fields in entries:
            res = envelope.parse_res(fields)
            cam = res["camera_id"]
            d = res["result"]
            prob = d["prob"]
            now = time.time()
            positive = prob >= config.VIO_THRESHOLD and d["num_persons"] >= config.MIN_PERSONS

            with _state_lock:
                _live[cam] = {"prob": prob, "num_persons": d["num_persons"], "ts": d["ts"]}
                w = _windows[cam]
                w.append(1 if positive else 0)
                fire = (sum(w) >= config.ALERT_K
                        and now - _last_alert.get(cam, 0) >= config.COOLDOWN_S)
                alert = None
                if fire:
                    _last_alert[cam] = now
                    _alert_seq += 1
                    alert = {
                        "id": _alert_seq,
                        "camera_id": cam,
                        "time": datetime.fromtimestamp(d["ts"]).strftime("%H:%M:%S %d/%m"),
                        "prob": prob,
                        "num_persons": d["num_persons"],
                        "snapshot_url": _media_url(d.get("snapshot_key", "")),
                        "clip_url": _media_url(d.get("clip_key", "")),
                    }
                    _alerts.appendleft(alert)

            _broadcast({"type": "live", "camera_id": cam,
                        "prob": prob, "num_persons": d["num_persons"]})
            if alert:
                print(f"[dash] 🚨 ALERT {cam} prob={prob:.2f} persons={d['num_persons']}")
                _broadcast({"type": "alert", "alert": alert})
            r.xack(config.RES_STREAM, config.CG_DASH, eid)


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
    threading.Thread(target=_consume_loop, daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


@app.get("/api/state")
async def state():
    with _state_lock:
        return {
            "live": _live,
            "alerts": list(_alerts),
            "config": {"threshold": config.VIO_THRESHOLD, "k": config.ALERT_K,
                       "n": config.ALERT_N, "cooldown_s": config.COOLDOWN_S},
        }


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


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=config.DASH_HOST, port=config.DASH_PORT)


_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phát hiện bạo lực — Như Quỳnh</title>
<style>
 :root{--bg:#0f1115;--card:#1a1d24;--ok:#2ecc71;--warn:#f39c12;--bad:#e74c3c;--txt:#e6e6e6}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;
   background:var(--bg);color:var(--txt)}
 header{padding:14px 20px;background:#13161c;border-bottom:1px solid #262b34;display:flex;
   align-items:center;gap:12px} header h1{font-size:18px;margin:0}
 .dot{width:10px;height:10px;border-radius:50%;background:var(--ok)}
 .wrap{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px}
 .cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 .cam{background:var(--card);border-radius:10px;padding:14px;border:1px solid #262b34}
 .cam h3{margin:0 0 8px;font-size:15px} .bar{height:10px;background:#262b34;border-radius:6px;overflow:hidden}
 .bar>i{display:block;height:100%;width:0;background:var(--ok);transition:width .3s,background .3s}
 .meta{font-size:12px;color:#9aa4b2;margin-top:6px}
 .panel{background:var(--card);border-radius:10px;padding:14px;border:1px solid #262b34;
   max-height:calc(100vh - 100px);overflow:auto}
 .panel h2{font-size:16px;margin:0 0 10px}
 .alert{display:flex;gap:10px;padding:10px;border-radius:8px;background:#241a1a;
   border:1px solid #5a2a2a;margin-bottom:10px}
 .alert img{width:120px;height:70px;object-fit:cover;border-radius:6px;background:#000}
 .alert b{color:var(--bad)} .alert .t{font-size:12px;color:#9aa4b2}
 .alert a{color:#6cb2ff;font-size:12px}
 .empty{color:#6b7280;font-size:14px}
</style></head>
<body>
<header><span class="dot" id="conn"></span><h1>🛡️ Giám sát hành vi bạo lực — xã Như Quỳnh</h1>
 <span class="meta" id="cfg"></span></header>
<div class="wrap">
 <div><h2 style="margin:4px 0 10px">Camera</h2><div class="cams" id="cams"><span class="empty">Chưa có dữ liệu camera…</span></div></div>
 <div class="panel"><h2>🚨 Cảnh báo gần đây</h2><div id="alerts"><span class="empty">Chưa có cảnh báo.</span></div></div>
</div>
<script>
const camsEl=document.getElementById('cams'), alertsEl=document.getElementById('alerts');
const cams={}; let hasAlert=false;
function color(p){return p>=0.7?'var(--bad)':p>=0.4?'var(--warn)':'var(--ok)';}
function renderCam(id,p,np){
 if(!cams[id]){camsEl.querySelector('.empty')?.remove();
  const d=document.createElement('div');d.className='cam';
  d.innerHTML=`<h3>📹 ${id}</h3><div class="bar"><i></i></div><div class="meta"></div>`;
  camsEl.appendChild(d);cams[id]=d;}
 const d=cams[id], bar=d.querySelector('i');
 bar.style.width=Math.round(p*100)+'%';bar.style.background=color(p);
 d.querySelector('.meta').textContent=`Nguy cơ: ${(p*100).toFixed(0)}% · Số người: ${np}`;
}
function renderAlert(a){
 if(!hasAlert){alertsEl.innerHTML='';hasAlert=true;}
 const el=document.createElement('div');el.className='alert';
 el.innerHTML=`${a.snapshot_url?`<a href="${a.clip_url||a.snapshot_url}" target="_blank">
   <img src="${a.snapshot_url}"></a>`:''}
  <div><b>BẠO LỰC ${(a.prob*100).toFixed(0)}%</b><div class="t">📹 ${a.camera_id} · 👥 ${a.num_persons} người</div>
  <div class="t">⏱ ${a.time}</div>${a.clip_url?`<a href="${a.clip_url}" target="_blank">▶ Xem clip</a>`:''}</div>`;
 alertsEl.prepend(el);
}
fetch('/api/state').then(r=>r.json()).then(s=>{
 document.getElementById('cfg').textContent=
  `Ngưỡng ${(s.config.threshold*100)|0}% · ${s.config.k}/${s.config.n} cửa sổ · cooldown ${s.config.cooldown_s}s`;
 for(const[id,v]of Object.entries(s.live))renderCam(id,v.prob,v.num_persons);
 s.alerts.slice().reverse().forEach(renderAlert);
});
const es=new EventSource('/events');
es.onopen=()=>document.getElementById('conn').style.background='var(--ok)';
es.onerror=()=>document.getElementById('conn').style.background='var(--bad)';
es.onmessage=e=>{const m=JSON.parse(e.data);
 if(m.type==='live')renderCam(m.camera_id,m.prob,m.num_persons);
 else if(m.type==='alert'){renderAlert(m.alert);
  if(window.AudioContext){try{const c=new AudioContext();const o=c.createOscillator();
   o.connect(c.destination);o.frequency.value=880;o.start();setTimeout(()=>o.stop(),250);}catch(e){}}}
};
</script></body></html>"""


if __name__ == "__main__":
    main()
