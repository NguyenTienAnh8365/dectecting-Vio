"""Cấu hình pipeline violence detection — đọc từ env, default chạy được ngay.

Fork từ Remote Sensing/transport/config.py. Khác biệt cốt lõi: bài toán này TEMPORAL
(clip-level), không per-frame. Pipeline 3 chặng:

  grabber → [REQ_STREAM] → gate (YOLO person, stateless, consumer group)
          → [GATED_STREAM] → classifier (STATEFUL per-camera, in-order, 1 process)
          → [RES_STREAM] → dashboard (k-of-n + cooldown → alert + SSE)
"""
from __future__ import annotations

import os


def _load_dotenv() -> None:
    """Nạp `.env` ở gốc dự án (secrets không nằm trong code/git). Env set sẵn được ưu tiên."""
    root = os.path.join(os.path.dirname(__file__), os.pardir)
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k, v)


_load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- Kết nối ---------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://:changeme@localhost:6381/0")

# --- Streams / consumer group ----------------------------------------------
REQ_STREAM = os.getenv("REQ_STREAM", "vio:frames:ready")     # grabber → gate
GATED_STREAM = os.getenv("GATED_STREAM", "vio:frames:gated")  # gate → classifier
RES_STREAM = os.getenv("RES_STREAM", "vio:results")           # classifier → dashboard
DLQ_STREAM = os.getenv("DLQ_STREAM", "vio:dead-letter")
CG_GATE = os.getenv("CG_GATE", "cg:gate")
CG_CLS = os.getenv("CG_CLS", "cg:cls")
CG_DASH = os.getenv("CG_DASH", "cg:dash")

# --- Backpressure / batch (ring-buffer realtime, drop frame cũ) -------------
REQ_MAXLEN = _int("REQ_MAXLEN", 2000)
GATED_MAXLEN = _int("GATED_MAXLEN", 2000)
RES_MAXLEN = _int("RES_MAXLEN", 4000)
READ_COUNT = _int("READ_COUNT", 16)
READ_BLOCK_MS = _int("READ_BLOCK_MS", 20)
CLAIM_MIN_IDLE_MS = _int("CLAIM_MIN_IDLE_MS", 30000)

# --- Độ tin cậy: idempotency + dead-letter ---------------------------------
DONE_TTL_S = _int("DONE_TTL_S", 120)
MAX_DELIVERIES = _int("MAX_DELIVERIES", 5)

# --- Person gate (YOLO) -----------------------------------------------------
GATE_MODEL = os.getenv("GATE_MODEL", "yolo11n.pt")  # nhẹ, tự tải; đổi yolo26x nếu cần chính xác
GATE_IMGSZ = _int("GATE_IMGSZ", 640)
GATE_CONF = _float("GATE_CONF", 0.35)
GATE_DEVICE = os.getenv("GATE_DEVICE", "0")
GATE_HALF = os.getenv("GATE_HALF", "1") == "1"
PERSON_CLASS = _int("PERSON_CLASS", 0)  # COCO 'person'
# Số người tối thiểu trong cửa sổ để alert được kích hoạt (gate alert, không gate frame).
MIN_PERSONS = _int("MIN_PERSONS", 2)

# --- Classifier clip-level --------------------------------------------------
VIO_BACKBONE = os.getenv("VIO_BACKBONE", "r2plus1d_18")  # torchvision.models.video
VIO_CKPT = os.getenv("VIO_CKPT", "weights/vio_r2plus1d.pt")  # rỗng/không tồn tại → dummy mode
VIO_DEVICE = os.getenv("VIO_DEVICE", "0")
VIO_HALF = os.getenv("VIO_HALF", "1") == "1"
WIN_LEN = _int("WIN_LEN", 16)        # số frame mỗi clip
WIN_STRIDE = _int("WIN_STRIDE", 8)   # phân loại lại mỗi N frame mới (cửa sổ trượt)
CLIP_SIZE = _int("CLIP_SIZE", 112)   # cạnh ảnh đưa vào model (r2plus1d/mc3 = 112)
MODEL_VERSION = os.getenv("MODEL_VERSION", "untrained")

# --- Logic alert (dashboard): chống false-positive + spam ------------------
VIO_THRESHOLD = _float("VIO_THRESHOLD", 0.70)  # ngưỡng prob 1 cửa sổ tính là "dương"
ALERT_K = _int("ALERT_K", 3)         # cần K cửa sổ dương...
ALERT_N = _int("ALERT_N", 5)         # ...trong N cửa sổ gần nhất → alert
COOLDOWN_S = _int("COOLDOWN_S", 45)  # mỗi camera tối đa 1 alert / COOLDOWN_S giây

# --- Lưu bằng chứng (disk local, dashboard serve) --------------------------
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")
SAVE_PROB = _float("SAVE_PROB", 0.50)  # prob >= mức này thì classifier lưu snapshot+clip
MEDIA_TTL_DAYS = _int("MEDIA_TTL_DAYS", 3)

# --- Dashboard --------------------------------------------------------------
DASH_HOST = os.getenv("DASH_HOST", "0.0.0.0")
DASH_PORT = _int("DASH_PORT", 8080)
