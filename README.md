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
| **mvit_v2_s** | phân loại **bạo lực** (chuyển động, clip 16f×224) | `mvit_v2_s-…` — Kinetics-400 | ✅ fine-tune 5 dataset (val F1 0.969) |

> **Đã đổi base model r2plus1d_18 → MViT v2-S** (video transformer mạnh hơn) + train production
> đa-dataset. MViT cần input **16 frame × 224×224**, norm Kinetics [0.45]/[0.225] (CỐ ĐỊNH theo
> positional-embed). YOLO gate vẫn build **TensorRT FP16** (`yolo26x.engine`). **MViT KHÔNG export
> được TRT** (op `SequenceAt/Empty/Insert` của pooling-attention, parser TRT 11.x không hỗ trợ) →
> classifier chạy **torch-FP16** từ `weights/vio_mvit_v2_s.pt` (vẫn nhanh trên Blackwell). Checkpoint
> r2plus1d cũ + engine giữ làm fallback.

## Cài đặt (server Blackwell — dùng `uv`)

```bash
uv venv && uv pip install -r requirements.txt   # .venv + deps (torch cu128 cho Blackwell)
cp .env.example .env           # sửa REDIS_PASSWORD, trỏ VIO_ENGINE/GATE_MODEL nếu dùng engine
docker compose up -d           # Redis ở cổng 6381
uv run python -m transport.init_streams
```

> Chưa có checkpoint/engine? Pipeline vẫn chạy thông ở **dummy mode** (prob=0, không bắn
> alert) để test transport/dashboard trước khi train xong.

## 1) Data & fine-tune (production, 5 dataset đa-domain)

| Dataset | Domain | Nguồn auth-free | Vai trò |
|---|---|---|---|
| **RLVS** (~2.3k) | street-fight cận cảnh | HF `khoipd/Violence` | train/val |
| **RWF-2000** (~2k) | CCTV giám sát | HF `DanJoshua/RWF-2000` (tar.gz) | train/val |
| **SCVD** (~480) | smart-city CCTV | HF `henriquequeirozcunha/scvd-violence-5fps` (parquet→mp4) | train/val |
| **Surveillance Fight** (300) | CCTV công cộng | GitHub `sayibet/fight-detection-surv-dataset` | train/val |
| **VioPeru** (367) | **CCTV thật Peru** | GitHub `hhuillcen/VioPeru` | **test cross-domain (giữ riêng)** |

SCVD bỏ lớp **Weaponized** (súng/dao tĩnh ít motion → nhiễu nhãn cho 3D/transformer). Tải:
`scripts/download_datasets.sh` + `scripts/build_scvd_clips.py`.

```bash
uv run python -m train.build_manifest        # → data/manifest.csv (path,label,dataset,split)
uv run python -m train.precompute_boxes       # cache crop-box người (crop-in-training, khớp infer)
uv run python -m train.train_clip --backbone mvit_v2_s --epochs 60 --batch 16 \
    --out weights/vio_mvit_v2_s.pt            # AMP + warmup→cosine + early-stop + class-weight
uv run python -m train.eval_test --split test           # đo cross-domain trên VioPeru
uv run python -m train.eval_threshold --manifest data/manifest.csv --split test  # tune ngưỡng kép
```

**Kết quả** (MViT v2-S, crop-in-training, span 3s): val (4 bộ in-domain) **F1 0.969 / AUC 0.994**;
test VioPeru (CCTV thật chưa train) clip-level **F1 0.73 / AUC 0.88**; verdict sliding-window
**P≈0.90 R≈0.56** (recall trần ~0.60 do domain-gap — muốn vượt cần footage CCTV địa phương fine-tune).

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

## Trạng thái tối ưu (production)

**✅ Đã làm** (đợt MViT production):
1. ✅ **Crop-to-person TRONG training** — `train/precompute_boxes.py` cache crop-box người (YOLO) →
   `train/train_clip.py` crop y như inference. Hết train/serve skew không-gian.
2. ✅ **Dữ liệu đa-domain** — 5 dataset (RLVS + RWF-2000 + SCVD + Surveillance Fight, VioPeru làm test).
3. ✅ **Quy trình train chuẩn** — AMP, AdamW param-groups (LR thấp backbone/cao head), warmup→cosine,
   early-stopping theo val-F1, class-weighted CE, augmentation (RRC/flip/color), tùy chọn mixup.
4. ✅ **MULTI-SPAN × crop/full** — infer lấy MAX prob qua `WIN_SECONDS_LIST=[3,7]` × {crop-người,
   full-frame} (`multispan_predict`). Sửa train/serve skew THỜI GIAN (16 frame liên tiếp ~0.5s →
   trải đều nhiều giây) + bắt cả đánh nhau **nhanh lẫn chậm/đám đông** (vd fight dashcam ~7s mù với
   span 3s) và **cảnh đông** (crop union ≈ cả khung → cần nhánh full-frame). VioPeru recall 0.34→**0.66**.
6. ✅ **Căn `torch`/`torchvision`** (2.10.0 ↔ 0.25.0+cu128) — hết lỗi `nms`. Preprocess backbone-aware
   (`BACKBONE_SPEC` + `frames_to_clip` dùng chung train+infer).

**⏳ Còn lại (ưu tiên):**
- **Footage CCTV địa phương (Như Quỳnh)** — recall cross-domain trần ~0.60 trên VioPeru; fine-tune
  thêm clip thật của xã là cách hiệu quả nhất để vượt (VioPeru hiện giữ làm test, không train).
- **Nhận diện vũ khí (súng/dao)** — classifier chỉ bắt *chuyển động*, mù với **súng tĩnh**. Cần YOLO
  vũ khí (Sohas/Roboflow) làm đường cảnh báo song song (đã chủ động loại SCVD-Weaponized khỏi train).
- **GPU/throughput** — MViT KHÔNG ra TRT được (op SequenceAt…); đang torch-FP16. Có thể thử torch.compile /
  batch nhiều cửa sổ; decode NVDEC thay cv2-CPU (đang nghẽn decode khi analyze nhiều video).
- **Overlay 3 mức** (Bình thường / Nghi ngờ / Bạo lực) cho khớp verdict ngưỡng kép.

## Pháp lý
Giám sát công + xử lý hình ảnh người dân → **Nghị định 13/2023/NĐ-CP** áp dụng. Đã có:
TTL tự xóa media (`MEDIA_TTL_DAYS`, mặc định 3 ngày) + human-in-the-loop + Basic Auth dashboard.
Cần bổ sung cho production: phân quyền truy cập chi tiết, policy lưu trữ/xóa, nhật ký truy cập.

## Đường lên edge (sau demo)
Stage **gate** và **classifier** stateless về contract (chỉ giao tiếp qua Redis) → bê thẳng lên Jetson:
export model TensorRT (gate `yolo … format=engine`; classifier ONNX→TRT — đã có), giữ nguyên
stream/dashboard. Server Blackwell làm "lò luyện" fine-tune + distill.
