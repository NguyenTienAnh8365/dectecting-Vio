# Phát hiện bạo lực realtime từ camera — demo xã Như Quỳnh

Phát hiện hành vi **gây gổ / xô đẩy / đánh nhau** từ camera, **sáng đèn dashboard +
snapshot/clip** cho CA xã xác nhận và có mặt kịp thời. Human-in-the-loop: hệ thống là
**decision-support**, người trực gật đầu cuối cùng — KHÔNG tự động điều người.

Camera nối **trực tiếp** vào dashboard.app — 1 tiến trình duy nhất, không qua message bus.
Khác biệt cốt lõi: bài toán này **temporal (clip-level)**, không per-frame → thêm 1 stage
**stateful per-camera** (cửa sổ trượt + video transformer).

> Bản đầu dùng Redis Streams (kế thừa từ dự án Remote Sensing — nhiều worker stateless chia 1
> hàng đợi). Đã bỏ hẳn: với vài camera IP nối thẳng 1 server AI, Redis chỉ tổ thêm JPEG
> encode/decode + round-trip mỗi frame, và khiến backlog tích lũy thay vì tự bỏ frame cũ khi xử
> lý không kịp (xem lịch sử git + docstring [transport/live.py](transport/live.py)).

## Hai luồng

**1. Live (RTSP/camera) — nhúng trong dashboard.app:**
```
RTSP/clip ─capture thread (1/camera, giữ đúng frame mới nhất)─>
  gate (YOLO person) ─classifier (STATEFUL per-camera, cửa sổ 16f, video transformer)─>
    callback trực tiếp trong process ─dashboard (ngưỡng kép + cooldown)─> 🔔 SSE + snapshot/clip (disk)
```

**2. Upload (tải video lên xem lại có detection):**
```
video upload ─> dashboard /api/upload ─> analyze.py (YOLO gate + classifier) ─>
  MP4 H.264 có overlay bbox + % bạo lực + verdict ─> xem lại trên dashboard
```

| Stage | File | Vai trò |
|---|---|---|
| live (camera) | [transport/live.py](transport/live.py) | capture thread/camera + gate + classifier, nhúng trong dashboard |
| model | [transport/model.py](transport/model.py) | torchvision video (MViT v2-S) + TensorRT runner |
| analyze | [transport/analyze.py](transport/analyze.py) | phân tích offline 1 video upload → MP4 có overlay |
| dashboard | [dashboard/app.py](dashboard/app.py) | chạy live.py + ngưỡng kép/cooldown → alert + SSE + upload |
| store | [transport/store.py](transport/store.py) | snapshot + clip bằng chứng (disk) |

## Hai mạng "cạnh nhau" — kế thừa checkpoint, KHÔNG train từ đầu

| Mạng | Vai trò | Kế thừa từ | Train thêm? |
|---|---|---|---|
| **YOLO** (yolo26x) | định vị **người** (không gian) | `yolo26x.pt` — COCO pretrained | ❌ COCO đã có "person" |
| **mvit_v2_s** | phân loại **bạo lực** (chuyển động, clip 16f×224) | `mvit_v2_s-…` — Kinetics-400 | ✅ fine-tune 5 dataset (val F1 0.969) |

> **Đã đổi base model r2plus1d_18 → MViT v2-S** (video transformer mạnh hơn) + train production
> đa-dataset. MViT cần input **16 frame × 224×224**, norm Kinetics [0.45]/[0.225] (CỐ ĐỊNH theo
> positional-embed). YOLO gate build **TensorRT FP16**. **MViT ĐÃ export được TRT** (dùng dynamo
> exporter — bản TorchScript cũ sinh op `SequenceAt/…` mới fail, xem [train/export_trt.py](train/export_trt.py);
> parity |Δprob| ≤ 0.001) NHƯNG trên GPU lớp Ampere TRT **không nhanh hơn torch-FP16** (batch-1
> 1.05x, batch-4 0.80x, dynamic-batch 0.5x — đo 3060 2026-07-02) → classifier live vẫn chạy
> **torch-FP16** từ `weights/vio_mvit_v2_s.pt`; engine chỉ đáng bật trên Jetson (đo lại trên máy
> đích). Mọi file `.engine` gắn chặt GPU build — build lại trên máy triển khai, không copy chéo.

## Capacity thực tế @30fps (đo 2026-07-02, RTX 3060 12GB ≈ A2000 12GB; A2000 chậm hơn ~25%)

| Thành phần | ms | Ghi chú |
|---|---|---|
| Gate yolo26x TRT-FP16 | 13.0/frame | 194 GFLOPs — quá to cho bài "chỉ tìm người" |
| Gate **yolo26m** TRT-FP16 | **7.4/frame** | khuyến nghị: chạy nổi 30fps thật |
| Gate yolo26s TRT-FP16 | 5.3/frame | nếu cần thêm cam (đổi lấy sót người nhỏ/xa) |
| MViT multispan (batch 4: 2 span × full+crop) | 156/lần | torch-FP16 |

Cấu hình chốt (`GRAB_FPS=30`, `WIN_STRIDE=15` → classify mỗi 0.5s; `ALERT_K=4/N=7` = 2s dương
trong 3.5s — đổi stride phải tính lại K/N theo công thức trong config.py; gate yolo26m): 1 cam
**có người** ≈ 534ms GPU/giây (gate 221 + MViT 313) → **3060 ~2 cam, A2000 ~1.5 cam "nóng"
đồng thời** (pipeline 2 tầng gate-thread + `LIVE_CLASSIFY_WORKERS=2` chồng preprocess/forward,
đo +23% throughput so với vòng tuần tự); cam **vắng người** chỉ tốn gate (221ms) nhờ
classify-skip → A2000 kham **3-4 cam** nếu phần lớn thời gian vắng (CCTV thật). Muốn nhiều cam
hơn: `WIN_STRIDE=24` (+20%), gate yolo26s (+15%), hoặc nâng GPU (RTX 4000 Ada / 4070 Ti ≈ gấp
đôi A2000). VRAM không phải nút cổ chai (~3.7GB đo 4 cam) — nghẽn ở compute; RAM CPU
~600MB/cam (buffer 210 frame 720p).

## Cài đặt (server Blackwell — dùng `uv`)

```bash
uv venv && uv pip install -r requirements.txt   # .venv + deps (torch cu128 cho Blackwell)
cp .env.example .env           # điền LIVE_CAMS, trỏ VIO_ENGINE/GATE_MODEL nếu dùng engine
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

**Đợt mở rộng CCTV (kế hoạch 2026-07-02 — thu hẹp domain-gap, mục tiêu recall VioPeru 0.56→0.65+):**

| Dataset | Domain | Nguồn | Ghi chú chuẩn bị |
|---|---|---|---|
| **UBI-Fights** (1k video, 80h, nhãn theo frame) | CCTV giám sát thật | [socia-lab.di.ubi.pt/EventDetection](http://socia-lab.di.ubi.pt/EventDetection/) — tải trực tiếp | video dài → cắt clip 3-7s theo nhãn frame-level; đoạn normal cùng video làm negative |
| **UCF-Crime** (chỉ lấy Fighting/Assault/Abuse + Normal) | **CCTV cột thật — khớp Như Quỳnh nhất** | [crcv.ucf.edu/projects/real-world](https://www.crcv.ucf.edu/projects/real-world/) (Dropbox ~128GB; chỉ cần các lớp trên, không tải cả bộ) | video untrimmed → trích đoạn violence theo temporal annotation + sample Normal |
| **CCTV-Fights** (1k video, 280 CCTV thật) | CCTV đánh nhau thật | [rose1.ntu.edu.sg/dataset/cctvFights](https://rose1.ntu.edu.sg/dataset/cctvFights/) — **phải đăng ký + ký agreement, nộp sớm** | dùng cho đợt sau khi được duyệt |
| Bus Violence (1.4k clip, Zenodo) / XD-Violence (4.7k, trộn phim) | phụ | [zenodo 7044203](https://zenodo.org/records/7044203) / [roc-ng.github.io/XD-Violence](https://roc-ng.github.io/XD-Violence/) | ưu tiên thấp: bus = diễn + nền rung; XD nhãn yếu, lệch domain — nếu dùng chỉ lọc subset Fighting/Riot ngắn |

> **Cách train khi thêm bộ mới: GỘP vào manifest chung, train lại 1 lần từ Kinetics-pretrained**
> — KHÔNG fine-tune tuần tự trên riêng bộ mới (catastrophic forgetting: model lệch hẳn về domain
> mới, quên 4 bộ cũ). Quy trình: cắt clip → thêm vào `data/` đúng cấu trúc → `build_manifest`
> (thêm nguồn mới) → `precompute_boxes` (chỉ chạy phần clip mới) → `train_clip` từ đầu với đủ
> 6 bộ (class-weight tự cân bằng) → `eval_test` trên VioPeru để so domain-gap trước/sau —
> **VioPeru vẫn giữ nguyên là test, không đưa vào train** (thước đo cross-domain duy nhất;
> chỉ gộp vào train khi đã có bộ test footage Như Quỳnh thật thay thế).
> Negative "cùng bối cảnh" (đoạn normal cắt từ chính video có violence) là loại negative giá
> trị nhất — dạy model phân biệt tụ tập bình thường vs xô xát trên cùng một khung camera.

**KẾT QUẢ đợt mở rộng (đã chạy 2026-07-02/03 — kết luận: KHÔNG dùng, giữ checkpoint 5 bộ):**

| Biến thể (VioPeru verdict multispan) | P | R | F1 |
|---|---|---|---|
| **5 bộ `vio_mvit_v2_s.pt` (giữ production)** | **0.877** | **0.664** | **0.756** |
| 7 bộ warm-start từ ckpt cũ (`_7ds.pt`) | 0.811 | 0.614 | 0.699 |
| 7 bộ from-Kinetics, data thô (`_7ds_scratch.pt`) | 0.827 | 0.579 | 0.681 |
| 7 bộ from-Kinetics, data lọc nhãn (`_7ds_clean.pt`, val 0.905) | 0.773 | 0.607 | 0.680 |
| như trên + train tiếp (`_7ds_clean_v2.pt`, val 0.907) | 0.743 | 0.579 | 0.651 |
| ensemble max(cũ, lọc) | 0.798 | 0.593 | 0.680 |

- Dữ liệu + script UBI/UCF đã GỠ khỏi repo sau kết luận (lấy lại ở commit `da9d5e0` nếu cần:
  script tải mirror Kaggle/HF, script cắt clip, danh sách 1211 clip nhãn bẩn đã loại).
  **Bẫy nhãn đã gặp:** negative "cùng bối cảnh" của UBI/UCF lẫn bạo lực thật (44%/28% được ckpt
  cũ chấm prob>0.7) + lớp Abuse gần như tĩnh → audit bằng ckpt cũ rồi lọc; val 7-bộ lên 0.905
  nhưng VioPeru vẫn không cải thiện.
- **Thí nghiệm quyết định (finetune VioPeru theo cụm camera `and/sanj/tala`,
  `--vioperu-train-groups sanj,tala`):** camera QUEN (video lạ) recall **0.64→0.91**; camera LẠ
  (cụm `and` giữ ngoài) recall 0.46→0.43 (không ăn). → Domain-gap nằm ở TỪNG CAMERA, không phải
  "phong cách CCTV" chung: **thêm dataset ngoài không phá được trần recall camera lạ; thu footage
  từ chính camera triển khai (Như Quỳnh) rồi finetune là con đường duy nhất đã được chứng minh.**

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
PY="uv run python" scripts/run_demo.sh data/people_test.mp4 cam1   # dashboard.app + đẩy clip loop
# mở http://<server>:8080
```

Hoặc chạy tay (1 tiến trình duy nhất — camera nối trực tiếp vào dashboard):
```bash
LIVE_CAMS="cam1=data/people_test.mp4" LIVE_LOOP=1 uv run python -m dashboard.app
```

**Chỉ dashboard (upload + xem lại, không cần camera live):**
```bash
uv run python -m dashboard.app   # LIVE_CAMS rỗng → kéo video vào ô upload
```

**Camera thật** — nhiều camera cách nhau `;`, không sửa code:
```bash
LIVE_CAMS="cam1=rtsp://user:pass@ip:554/Streaming/Channels/101;cam2=rtsp://..." uv run python -m dashboard.app
```

**Test nhanh riêng pipeline camera** (in kết quả ra console, không cần dashboard):
```bash
uv run python -m transport.live --cam cam1=data/people_test.mp4 --loop
```

Dừng nền: `pkill -f 'dashboard\.app'`

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
`transport/live.py` không phụ thuộc hạ tầng server (không Redis/message bus) → bê thẳng lên
Jetson: export model TensorRT (gate `yolo … format=engine`; classifier ONNX→TRT — đã có), giữ
nguyên `dashboard/app.py`. Server Blackwell làm "lò luyện" fine-tune + distill.
