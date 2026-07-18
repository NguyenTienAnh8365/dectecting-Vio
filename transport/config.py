"""Cấu hình pipeline violence detection — đọc từ env, default chạy được ngay.

Khác biệt cốt lõi: bài toán này TEMPORAL (clip-level), không per-frame. Camera nối TRỰC TIẾP
vào dashboard.app (transport/live.py, KHÔNG qua message bus — xem live.py để hiểu lý do):

  camera (LIVE_CAMS) → gate (YOLO person) → classifier (STATEFUL per-camera, cửa sổ trượt)
                     → dashboard (k-of-n + cooldown → alert + SSE), CÙNG 1 tiến trình.
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


def _harden_console() -> None:
    """Console Windows có thể trả OSError WinError 1 khi write (console mode bị tiến trình
    khác đổi) — một lệnh print không được phép giết thread pipeline."""
    import io
    import sys

    class _SafeWriter(io.TextIOBase):
        def __init__(self, raw):
            self._raw = raw

        def write(self, s: str) -> int:
            try:
                return self._raw.write(s)
            except OSError:
                return len(s)

        def flush(self) -> None:
            try:
                self._raw.flush()
            except OSError:
                pass

        @property
        def encoding(self):
            return getattr(self._raw, "encoding", "utf-8")

    for name in ("stdout", "stderr"):
        raw = getattr(sys, name)
        if raw is not None:
            setattr(sys, name, _SafeWriter(raw))


_load_dotenv()
if os.name == "nt":
    _harden_console()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- Camera live -------------------------------------------------------------
# "cam1=rtsp://user:pass@ip/stream;cam2=data/fight.mp4" — dashboard.app tự chạy transport.live
# nhúng trong tiến trình (thread), rỗng = không có camera live (chỉ upload). Xem transport/live.py.
LIVE_CAMS = os.getenv("LIVE_CAMS", "")
# Nguồn FILE hết clip thì tua lại (demo bằng 1 clip lặp vô hạn) — camera rtsp/http bỏ qua cờ này
# (luôn tự nhịp theo camera, không có khái niệm "hết clip").
LIVE_LOOP = os.getenv("LIVE_LOOP", "0") == "1"

# --- Person gate (YOLO) -----------------------------------------------------
GATE_MODEL = os.getenv("GATE_MODEL", "yolo11n.pt")  # nhẹ, tự tải; đổi yolo26x nếu cần chính xác
GATE_IMGSZ = _int("GATE_IMGSZ", 640)
GATE_CONF = _float("GATE_CONF", 0.35)
GATE_DEVICE = os.getenv("GATE_DEVICE", "0")
GATE_HALF = os.getenv("GATE_HALF", "1") == "1"
# ultralytics mới bỏ tham số predict 'half' → dùng 'quantize' (16=FP16, None=FP32). Map sẵn.
GATE_QUANTIZE = 16 if GATE_HALF else None
PERSON_CLASS = _int("PERSON_CLASS", 0)  # COCO 'person'
# yolo26x.engine build BATCH TĨNH (=1) — transport/live.py luôn gọi predict() đúng 1 ảnh/lần nên
# tự nhiên không bao giờ vượt batch engine (không cần chunk như bản Redis cũ).
# Số người tối thiểu trong cửa sổ để alert được kích hoạt (gate alert, không gate frame).
MIN_PERSONS = _int("MIN_PERSONS", 2)
# Phân tích video upload: chỉ chạy YOLO mỗi N frame (tái dùng bbox cho overlay giữa các lần)
# → giảm số lần infer gate, nhanh hơn. Vẫn ép chạy gate đúng frame quyết định cửa sổ để
# đếm người chính xác. Tăng lên 3-4 nếu cần nhanh hơn nữa (bbox trễ tối đa N frame).
ANALYZE_GATE_EVERY = _int("ANALYZE_GATE_EVERY", 3)
# Live: cùng trick — YOLO mỗi N frame, frame giữa tái dùng bbox cũ (người không dịch đáng kể
# trong ~0.25s @ 12fps). Đo 3060 2026-07-02: gate yolo26x 17.4ms/frame = 46% GPU/camera →
# N=3 cắt ~2/3 phần đó. Vẫn ép gate đúng frame QUYẾT ĐỊNH cửa sổ (đếm người chuẩn cho alert).
LIVE_GATE_EVERY = _int("LIVE_GATE_EVERY", 3)
# Bỏ classify khi cả cửa sổ KHÔNG có người (max persons < mức này) — camera vắng (phần lớn
# thời gian CCTV thật) chỉ tốn gate, không tốn MViT (166ms/lần trên 3060). Alert vốn đã cần
# MIN_PERSONS≥2 nên bỏ classify lúc 0 người không thể làm mất alert; để 1 cho an toàn
# (YOLO sót người 1 vài frame vẫn còn các frame khác trong cửa sổ giữ classify chạy).
CLASSIFY_MIN_PERSONS = _int("CLASSIFY_MIN_PERSONS", 1)
# Số worker classify chạy SONG SONG với gate thread (pipeline 2 tầng). 1 thread tuần tự
# gate→preprocess→forward để GPU nghỉ lúc CPU làm việc (đo 4 cam 2026-07-02: GPU util chỉ
# ~83%). torch forward/cv2 nhả GIL nên threading đủ (không cần process). 2 worker cho phép
# preprocess clip này chồng lên forward clip kia. Job per-cam kiểu mailbox: cửa sổ mới nhất
# thắng, không tích backlog (đúng tinh thần FrameBox).
LIVE_CLASSIFY_WORKERS = _int("LIVE_CLASSIFY_WORKERS", 2)

# --- Classifier clip-level --------------------------------------------------
VIO_BACKBONE = os.getenv("VIO_BACKBONE", "mvit_v2_s")  # torchvision.models.video (đã đổi từ r2plus1d_18)
VIO_ENGINE = os.getenv("VIO_ENGINE", "")  # .engine TensorRT FP16 (ưu tiên nếu có) → full-TRT
VIO_CKPT = os.getenv("VIO_CKPT", "weights/vio_r2plus1d.pt")  # rỗng/không tồn tại → dummy mode
VIO_DEVICE = os.getenv("VIO_DEVICE", "0")
VIO_HALF = os.getenv("VIO_HALF", "1") == "1"
WIN_LEN = _int("WIN_LEN", 16)        # số frame ĐƯA VÀO model (mvit CỐ ĐỊNH 16)
# Số frame giữa 2 lần classify. @GRAB_FPS=30: 15 frame = chấm mỗi 0.5s (chốt 2026-07-02).
# Chấm dày hơn KHÔNG tăng khả năng phát hiện: mỗi lần chấm model nhìn cả cửa sổ 3s/7s lịch sử,
# trượt 0.5s/bước nên cú đánh 1s vẫn nằm trong ~6-14 lần chấm liên tiếp; đường strong nổ ≤0.5s.
# Mỗi lần chấm ~156ms GPU → stride là núm vặn compute số 1 (15 → 2 lần/s/cam có người).
# ALERT_K/N phải chỉnh THEO stride để giữ ngữ nghĩa thời gian (xem dưới).
WIN_STRIDE = _int("WIN_STRIDE", 15)
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
# classifier.py sống rất lâu (process live liên tục) — batch multispan đổi theo cảnh (đông người
# → crop+full, vắng → chỉ full) làm PyTorch CUDA allocator giữ mốc VRAM cao nhất từng cấp phát,
# cộng dồn qua thời gian (đo thực tế 2026-07-01, xem transport/analyze.py). Dọn định kỳ theo giờ
# đồng hồ (không phải mỗi lần classify — empty_cache() đồng bộ hoá, gọi liên tục hại latency live).
EMPTY_CACHE_INTERVAL_S = _int("EMPTY_CACHE_INTERVAL_S", 300)
MODEL_VERSION = os.getenv("MODEL_VERSION", "untrained")

# --- Logic alert (dashboard): chống false-positive + spam ------------------
# Ngưỡng KÉP theo độ tin cậy (tránh bỏ sót hành động bạo lực ngắn 1-2 cửa sổ):
#   • đường KÉO DÀI: cửa sổ "dương" khi prob ≥ VIO_THRESHOLD; cần K/N dương → alert.
#     Bắt bạo lực độ tin cậy vừa nhưng liên tục, đồng thời lọc nhiễu lẻ (chống FP).
#   • đường MẠNH:    chỉ 1 cửa sổ prob ≥ VIO_THRESHOLD_STRONG đã đủ alert ngay, KHÔNG
#     cần đủ K/N. Bắt cú đánh/tát chớp nhoáng — "1 hành động nhỏ cũng là bạo lực".
# => verdict = (đủ K/N ở mức thấp) OR (có ≥1 cửa sổ ở mức mạnh). Cả hai vẫn qua gate người.
VIO_THRESHOLD = _float("VIO_THRESHOLD", 0.50)         # ngưỡng "dương" cho đường kéo dài
VIO_THRESHOLD_STRONG = _float("VIO_THRESHOLD_STRONG", 0.80)  # cửa sổ "mạnh" khi prob ≥ mức này
# Đường mạnh cần STREAK cửa sổ mạnh LIÊN TIẾP (không phải 1): spike nhiễu 1 cửa sổ (lắc cam,
# đứng lên ngồi xuống đột ngột — chuyển động toàn khung giống bạo lực với model) chết hẳn,
# còn cú đánh thật dù chớp nhoáng vẫn nằm trong cửa sổ trượt 3s/7s của ~6 lần chấm liên tiếp
# → prob cao ở nhiều cửa sổ liền nhau, chỉ trễ thêm 1 nhịp chấm (~0.5s), KHÔNG mất phát hiện.
VIO_STRONG_STREAK = _int("VIO_STRONG_STREAK", 2)
# K/N định nghĩa theo SỐ CỬA SỔ nên ngữ nghĩa THỜI GIAN phụ thuộc nhịp chấm (WIN_STRIDE/GRAB_FPS).
# Thiết kế gốc (tune trên VioPeru): ~2s dương trong ~3.3s. @30fps stride 15 (chấm mỗi 0.5s):
# K=4/N=7 = 2s dương trong 3.5s — GIỮ NGUYÊN ngữ nghĩa đó. Đổi stride thì tính lại:
# K ≈ 2s/(stride/fps), N ≈ 3.5s/(stride/fps).
ALERT_K = _int("ALERT_K", 4)         # cần K cửa sổ dương...
ALERT_N = _int("ALERT_N", 7)         # ...trong N cửa sổ gần nhất → alert
COOLDOWN_S = _int("COOLDOWN_S", 45)  # mỗi camera tối đa 1 alert / COOLDOWN_S giây

# --- Lưu bằng chứng (disk local, dashboard serve) --------------------------
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")
# prob >= mức này thì classifier lưu snapshot+clip. RÀNG BUỘC: phải ≤ VIO_THRESHOLD — alert
# k-of-n mượn snapshot của cửa sổ DƯƠNG gần nhất (dashboard._handle_live_result), nếu cửa sổ
# dương mà không lưu media thì alert trắng ảnh. Kẹp cứng để không ai chỉnh lệch được.
SAVE_PROB = min(_float("SAVE_PROB", 0.50), VIO_THRESHOLD)
# Độ dài clip bằng chứng (giây, NHỊP THẬT — lấy các frame cuối buffer theo timestamp).
# Trước đây lưu 16 frame trải đều cả buffer 7s phát ở 12fps → clip ~1.3s kiểu tua nhanh, khó xem.
EVIDENCE_CLIP_S = _float("EVIDENCE_CLIP_S", 3.0)
MEDIA_TTL_DAYS = _int("MEDIA_TTL_DAYS", 3)

# --- Dashboard --------------------------------------------------------------
DASH_HOST = os.getenv("DASH_HOST", "0.0.0.0")
DASH_PORT = _int("DASH_PORT", 8080)
