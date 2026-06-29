# Phát hiện bạo lực realtime từ camera — demo xã Như Quỳnh

Phát hiện hành vi **gây gổ / xô đẩy / đánh nhau** từ camera, **sáng đèn dashboard +
snapshot/clip** cho CA xã xác nhận và có mặt kịp thời. Human-in-the-loop: hệ thống là
**decision-support**, người trực gật đầu cuối cùng — KHÔNG tự động điều người.

Tái dùng tầng transport (Redis Streams + YOLO + reliability) của dự án **Remote Sensing**.
Khác biệt cốt lõi: bài toán này **temporal (clip-level)**, không per-frame → thêm 1 stage
**stateful per-camera** (cửa sổ trượt + 3D-CNN).

## Hai luồng

**1. Live (RTSP/camera):**
```
RTSP/clip ─grabber─> [vio:frames:ready] ─gate (YOLO person, stateless, group)─>
  [vio:frames:gated] ─classifier (STATEFUL per-camera, cửa sổ 16f, 3D-CNN)─>
    [vio:results] ─dashboard (ngưỡng kép + cooldown)─> 🔔 SSE + snapshot/clip (disk)
```

**2. Upload (tải video lên xem lại có detection):**
```
video upload ─> dashboard /api/upload ─> analyze.py (YOLO gate + classifier) ─>
  MP4 H.264 có overlay bbox + % bạo lực + verdict ─> xem lại trên dashboard
```

| Stage | File | Tái dùng từ RS | Vai trò |
|---|---|---|---|
| grabber | [transport/grabber.py](transport/grabber.py) | producer.py | clip/RTSP → frame stream |
| person-gate | [transport/gate.py](transport/gate.py) | worker.py | YOLO đếm người + bbox (stateless, group) |
| classifier | [transport/classifier.py](transport/classifier.py) | 🆕 mới | cửa sổ trượt per-camera → prob bạo lực |
| model | [transport/model.py](transport/model.py) | 🆕 mới | torchvision video (r2plus1d_18) + TensorRT runner |
| analyze | [transport/analyze.py](transport/analyze.py) | 🆕 mới | phân tích offline 1 video upload → MP4 có overlay |
| dashboard | [dashboard/app.py](dashboard/app.py) | consumer.py | ngưỡng kép + cooldown → alert + SSE + upload |
| store | [transport/store.py](transport/store.py) | storage.py (MinIO→disk) | snapshot + clip bằng chứng |

## Hai mạng "cạnh nhau" — kế thừa checkpoint, KHÔNG train từ đầu

| Mạng | Vai trò | Kế thừa từ | Train thêm? |
|---|---|---|---|
| **YOLO** (yolo26x) | định vị **người** (không gian) | `yolo26x.pt` — COCO pretrained | ❌ COCO đã có "person" |
| **r2plus1d_18** | phân loại **bạo lực** (chuyển động, clip 16f) | `r2plus1d_18-91a641e6.pth` — Kinetics-400 | ✅ fine-tune RLVS (97.7%) |

Cả hai đều build **TensorRT FP16 engine** để chạy nhanh (`yolo26x.engine`, `vio_r2plus1d_fp16.engine`).

## Cài đặt (server Blackwell — dùng `uv`)

```bash
uv venv && uv pip install -r requirements.txt   # .venv + deps (torch cu128 cho Blackwell)
cp .env.example .env           # sửa REDIS_PASSWORD, trỏ VIO_ENGINE/GATE_MODEL nếu dùng engine
docker compose up -d           # Redis ở cổng 6381
uv run python -m transport.init_streams
```

> Chưa có checkpoint/engine? Pipeline vẫn chạy thông ở **dummy mode** (prob=0, không bắn
> alert) để test transport/dashboard trước khi train xong.

## 1) Data & fine-tune

- **RLVS** (Real Life Violence Situations, ~2.3k clip) — tải auth-free từ HuggingFace
  `khoipd/Violence`, giải nén `data/RLVS/Real_Life_Violence_Dataset/{Violence,NonViolence}`.
- **RWF-2000** (CCTV, đúng domain giám sát nhất) — xin form GitHub `mchengny/RWF2000` (cho tối ưu sau).

```bash
uv run python -m train.train_clip --data data/RLVS --backbone r2plus1d_18 \
    --epochs 8 --batch 8 --out weights/vio_r2plus1d.pt
# build TensorRT FP16 (dùng default paths):  uv run python -m train.export_trt  → weights/vio_r2plus1d_fp16.engine
```

## 2) Chạy demo end-to-end

```bash
docker compose up -d
PY="uv run python" scripts/run_demo.sh data/people_test.mp4 cam1   # dựng 4 tiến trình + đẩy clip loop
# mở http://<server>:8080
```

Hoặc chạy tay từng tiến trình (mỗi cái 1 terminal):
```bash
uv run python -m transport.init_streams
uv run python -m transport.gate
uv run python -m transport.classifier
uv run python -m dashboard.app
uv run python -m transport.grabber --src data/people_test.mp4 --camera cam1 --fps 12 --loop
```

**Chỉ dashboard (upload + xem lại, không cần camera live):**
```bash
docker compose up -d && uv run python -m dashboard.app   # kéo video vào ô upload
```

**Camera thật** chỉ đổi `--src`, không sửa code:
```bash
uv run python -m transport.grabber --src "rtsp://user:pass@ip:554/Streaming/Channels/101" --camera cam1 --fps 12
```

Dừng nền: `pkill -f 'transport\.|dashboard\.app'`

## Logic phát hiện (các quyết định cốt lõi)

**Ngưỡng KÉP theo độ tin cậy** ([transport/config.py](transport/config.py)) — tránh bỏ sót hành động bạo lực ngắn:
```
verdict = (số cửa sổ prob ≥ VIO_THRESHOLD đạt K/N)   OR   (≥1 cửa sổ prob ≥ VIO_THRESHOLD_STRONG)
```
- Đường **kéo dài** (k-of-n): bạo lực độ tin cậy vừa nhưng liên tục; lọc nhiễu lẻ (chống FP).
- Đường **mạnh** (1 cửa sổ ≥ ngưỡng cao): bắt cú đánh/tát chớp nhoáng — "1 hành động nhỏ cũng là bạo lực".
- Cả hai vẫn qua gate `MIN_PERSONS`.

**Crop vùng người trước classifier** ([transport/model.py](transport/model.py) `person_crop_box`) — bù
domain gap: r2plus1d train trên RLVS (đánh nhau cận cảnh) nên với CCTV góc rộng (người nhỏ ở góc, co
cả khung về 112×112 mất tín hiệu) thì prob bị thấp. Cắt quanh union bbox người → hành động to gần kín
khung như RLVS. *Đo thực tế video432 CCTV: max 0.535 → 0.868; cảnh người đi bình thường giữ 0.227 (không FP).*

**Công cụ đo ngưỡng** [train/eval_threshold.py](train/eval_threshold.py) — chạy classifier trên video có
nhãn, sweep (T_low, T_strong, K) → Precision/Recall/F1 để chọn ngưỡng có cơ sở dữ liệu.

## Tinh chỉnh ([transport/config.py](transport/config.py))

| Nhóm | Biến | Mặc định | Ghi chú |
|---|---|---|---|
| Cửa sổ | `WIN_LEN` · `WIN_STRIDE` | 16 · 8 | clip 16 frame, phân loại lại mỗi 8 frame |
| Ngưỡng thấp | `VIO_THRESHOLD` | 0.50 | cửa sổ "dương" cho đường k-of-n |
| Ngưỡng mạnh | `VIO_THRESHOLD_STRONG` | 0.80 | 1 cửa sổ ≥ mức này → alert ngay |
| Smoothing | `ALERT_K` / `ALERT_N` | 3 / 5 | cần 3/5 cửa sổ dương → alert |
| Spam | `COOLDOWN_S` | 45 | mỗi camera tối đa 1 alert / 45s |
| Person-gate | `MIN_PERSONS` | 2 | <2 người thì không alert |
| Crop | `CROP_TO_PERSON` · `CROP_PAD` | 1 · 0.35 | cắt vùng người trước classifier (bù domain gap) |
| Tốc độ upload | `ANALYZE_GATE_EVERY` | 2 | chạy YOLO mỗi N frame khi phân tích video (tái dùng bbox) |
| Lưu | `SAVE_PROB` | 0.50 | prob ≥ mức này mới lưu snapshot+clip |

## ⚠️ Cần tối ưu sau này (hiện tại là baseline cho demo)

Training 8 epoch hiện tại chỉ là **baseline để test pipeline**. Khi tối ưu, theo thứ tự ưu tiên:

1. **Crop-to-person TRONG training** (ưu tiên #1) — inference đã crop vùng người nhưng
   [train/train_clip.py](train/train_clip.py) còn train trên **cả khung** → *train/serve skew*.
   Train lại có crop (dùng `person_crop_box`) để khớp phân phối → prob sẽ lên thêm.
2. **Thêm dữ liệu đúng domain** — fine-tune kèm **RWF-2000** (CCTV giám sát) hoặc clip CCTV thật
   của xã, không chỉ RLVS (street fight cận cảnh).
3. **Tối ưu quy trình train** — augmentation không-gian (random crop/flip/color), LR schedule
   (cosine + warmup, LR thấp cho backbone), early-stopping thay vì cố định 8 epoch, mixup/cutmix clip.
4. **Nhận diện vũ khí (súng/dao)** — r2plus1d chỉ bắt *chuyển động*, mù với **súng tĩnh** (vd cướp
   cửa hàng tiện lợi). Cần train thêm YOLO trên dataset vũ khí (Sohas/Roboflow) làm đường cảnh báo song song.
5. **GPU/throughput** — batch nhiều cửa sổ qua classifier engine khi có >1 camera; decode bằng NVDEC
   thay cho cv2 CPU (bỏ nghẽn decode trong analyze).
6. **Kỹ thuật** — căn chỉnh phiên bản `torch`/`torchvision` cho khớp (đang lệch 2.10/0.26, warning lúc
   import); làm overlay 3 mức (Bình thường / Nghi ngờ / Bạo lực) cho khớp verdict, hết mâu thuẫn hiển thị.

## Pháp lý
Giám sát công + xử lý hình ảnh người dân → **Nghị định 13/2023/NĐ-CP** áp dụng. Đã có:
TTL tự xóa media (`MEDIA_TTL_DAYS`, mặc định 3 ngày) + human-in-the-loop + Basic Auth dashboard.
Cần bổ sung cho production: phân quyền truy cập chi tiết, policy lưu trữ/xóa, nhật ký truy cập.

## Đường lên edge (sau demo)
Stage **gate** và **classifier** stateless về contract (chỉ giao tiếp qua Redis) → bê thẳng lên Jetson:
export model TensorRT (gate `yolo … format=engine`; classifier ONNX→TRT — đã có), giữ nguyên
stream/dashboard. Server Blackwell làm "lò luyện" fine-tune + distill.
