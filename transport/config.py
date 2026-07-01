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
# ultralytics mới bỏ tham số predict 'half' → dùng 'quantize' (16=FP16, None=FP32). Map sẵn.
GATE_QUANTIZE = 16 if GATE_HALF else None
PERSON_CLASS = _int("PERSON_CLASS", 0)  # COCO 'person'
# Số người tối thiểu trong cửa sổ để alert được kích hoạt (gate alert, không gate frame).
MIN_PERSONS = _int("MIN_PERSONS", 2)
# Phân tích video upload: chỉ chạy YOLO mỗi N frame (tái dùng bbox cho overlay giữa các lần)
# → giảm số lần infer gate, nhanh hơn. Vẫn ép chạy gate đúng frame quyết định cửa sổ để
# đếm người chính xác. Tăng lên 3-4 nếu cần nhanh hơn nữa (bbox trễ tối đa N frame).
ANALYZE_GATE_EVERY = _int("ANALYZE_GATE_EVERY", 3)

# --- Classifier clip-level --------------------------------------------------
VIO_BACKBONE = os.getenv("VIO_BACKBONE", "mvit_v2_s")  # torchvision.models.video (đã đổi từ r2plus1d_18)
VIO_ENGINE = os.getenv("VIO_ENGINE", "")  # .engine TensorRT FP16 (ưu tiên nếu có) → full-TRT
VIO_CKPT = os.getenv("VIO_CKPT", "weights/vio_r2plus1d.pt")  # rỗng/không tồn tại → dummy mode
VIO_DEVICE = os.getenv("VIO_DEVICE", "0")
VIO_HALF = os.getenv("VIO_HALF", "1") == "1"
WIN_LEN = _int("WIN_LEN", 16)        # số frame ĐƯA VÀO model (mvit CỐ ĐỊNH 16)
# 8 (đã từng đổi lên 16 để analyze nhanh hơn — xem log cũ): hạ lại còn 8 để giảm độ trễ phản
# ứng (chấm lại thường hơn = phát hiện sớm hơn). Chấp nhận được vì _PREP_POOL (threading resize
# CPU trong predict_batch) đã giảm ~1.8× thời gian mỗi lần classify, bù lại phần tăng do gọi
# nhiều lần hơn. Muốn trễ thấp hơn nữa thì hạ tiếp, đổi lại tốn compute hơn.
WIN_STRIDE = _int("WIN_STRIDE", 8)
# Span THỜI GIAN của 1 cửa sổ phân loại. Training lấy 16 frame TRẢI ĐỀU cả clip (~vài giây);
# nếu infer lấy 16 frame LIÊN TIẾP (ở fps gốc chỉ ~0.5s) thì lệch train/serve về thời gian →
# bỏ sót (đo trên VioPeru: recall 0.26 khi span 0.5s → 0.56 khi span ~3s). Nên: buffer
# round(WIN_SECONDS*fps) frame rồi lấy đều 16 frame. WIN_SECONDS≈3s khớp độ dài clip RLVS/RWF.
# MULTI-SPAN: phân loại ở NHIỀU span rồi lấy MAX (bắt cả đánh nhau nhanh ~3s lẫn chậm/đám đông ~7s;
# 1 span cố định không kham cả hai — vd fight dashcam diễn ra ~7s thì span 3s cho prob ~0). Buffer =
# span LỚN NHẤT. Đánh đổi: trễ verdict đầu = span lớn nhất (vài giây) — chấp nhận được với cảnh báo bạo lực.
WIN_SECONDS_LIST = [float(x) for x in os.getenv("WIN_SECONDS_LIST", "3.0,7.0").split(",")]
WIN_SECONDS = max(WIN_SECONDS_LIST)  # span lớn nhất (kích thước buffer)
GRAB_FPS = _int("GRAB_FPS", 12)      # fps luồng live (grabber) — để quy span ra số frame
WIN_SPAN = max(WIN_LEN, round(WIN_SECONDS * GRAB_FPS))  # số frame buffer ở classifier live
# Cạnh ảnh đưa vào model — mặc định SUY theo backbone (CNN video=112, mvit/transformer=224) để
# train/infer/engine không lệch. Nguồn chân lý là BACKBONE_SPEC ở transport/model.py; map dưới chỉ
# để default (tránh import vòng config↔model). Env CLIP_SIZE vẫn override nếu cần.
_DEFAULT_SIZE = {"r2plus1d_18": 112, "mc3_18": 112, "r3d_18": 112,
                 "s3d": 224, "mvit_v1_b": 224, "mvit_v2_s": 224}
CLIP_SIZE = _int("CLIP_SIZE", _DEFAULT_SIZE.get(VIO_BACKBONE, 112))
# Cắt quanh vùng người (union bbox YOLO) TRƯỚC khi đưa vào classifier → hành động bạo lực
# to gần kín khung như RLVS, bù domain gap với CCTV góc rộng (người nhỏ, co về 112 mất tín
# hiệu). Tắt bằng CROP_TO_PERSON=0. CROP_PAD = phần đệm quanh union (tỉ lệ cạnh).
CROP_TO_PERSON = os.getenv("CROP_TO_PERSON", "1") == "1"
CROP_PAD = _float("CROP_PAD", 0.35)
# Resize/normalize lúc INFER qua GPU tensor (F.interpolate mode="area", khớp thuật toán
# INTER_AREA của cv2) thay vì cv2.resize CPU tuần tự — nhanh hơn ~1.9x, đã validate KHÔNG giảm
# accuracy trên VioPeru test (F1 0.728→0.727, AUC 0.884→0.893, đo 2026-07-01, xem
# transport/model.py:frames_to_clip_gpu). CHỈ áp dụng lúc infer — training (train_clip.py) vẫn
# giữ nguyên cv2 CPU (DataLoader multi-process không hợp CUDA). Tắt về cv2 CPU bằng =0.
PREP_GPU_RESIZE = os.getenv("PREP_GPU_RESIZE", "1") == "1"
MODEL_VERSION = os.getenv("MODEL_VERSION", "untrained")

# --- Logic alert (dashboard): chống false-positive + spam ------------------
# Ngưỡng KÉP theo độ tin cậy (tránh bỏ sót hành động bạo lực ngắn 1-2 cửa sổ):
#   • đường KÉO DÀI: cửa sổ "dương" khi prob ≥ VIO_THRESHOLD; cần K/N dương → alert.
#     Bắt bạo lực độ tin cậy vừa nhưng liên tục, đồng thời lọc nhiễu lẻ (chống FP).
#   • đường MẠNH:    chỉ 1 cửa sổ prob ≥ VIO_THRESHOLD_STRONG đã đủ alert ngay, KHÔNG
#     cần đủ K/N. Bắt cú đánh/tát chớp nhoáng — "1 hành động nhỏ cũng là bạo lực".
# => verdict = (đủ K/N ở mức thấp) OR (có ≥1 cửa sổ ở mức mạnh). Cả hai vẫn qua gate người.
VIO_THRESHOLD = _float("VIO_THRESHOLD", 0.50)         # ngưỡng "dương" cho đường kéo dài
VIO_THRESHOLD_STRONG = _float("VIO_THRESHOLD_STRONG", 0.80)  # 1 cửa sổ ≥ mức này → alert ngay
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
