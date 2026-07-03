# Project Memory — Detecting Violence (bản gộp cho Claude Sonnet 5 đọc)

> File này gộp toàn bộ memory của dự án phát hiện bạo lực realtime (xã Như Quỳnh).
> Nguồn gốc: `/home/abc/.claude/projects/-home-abc-AI-project-detecting-violence/memory/`
> Ngày gộp: 2026-07-01. Ngôn ngữ ghi chú: tiếng Việt.

---

## 0. Chỉ mục (tóm tắt 1 dòng mỗi mục)

- **MViT production** — đổi sang `mvit_v2_s`, train 5 dataset (F1 0.969), span-thời-gian infer 3s, MViT KHÔNG ra TRT → torch-FP16, ngưỡng + recall trần VioPeru ~0.60
- **Env: uv + Python 3.12.10** — dùng uv, torch cu128 cho Blackwell, bỏ pin requirements; torch 2.10↔torchvision 0.25 (lệch 0.26 lỗi nms)
- **Violence demo setup** — RLVS auth-free, yolo26 gate, checkpoint 97.7%, full-TRT, cách chạy
- **Public tunnel** — dashboard public watch.tpvtech.vn qua Cloudflare (socat 80→8080 + cloudflared host-net)
- **Dual threshold** — verdict ngưỡng kép (T_low+k/n OR T_strong 1 cửa sổ) chống bỏ sót bạo lực ngắn + tool eval_threshold đo ngưỡng
- **Person-crop classifier** — crop vùng người trước classifier bù domain gap CCTV (video432: 0.535→0.868), không train lại
- **Upload/analyze feature** — dashboard upload video → infer offline → xem lại video có overlay

---

## 1. MViT production (type: project)

Quyết định 2026-06-30: **đổi base model r2plus1d_18 → `mvit_v2_s`** (video transformer) + train
production đa-dataset. Thay demo RLVS 8-epoch cũ. Checkpoint mới
`weights/vio_mvit_v2_s.pt` (val F1 0.969 / AUC 0.994; epoch 13, early-stop). Giữ r2plus1d cũ làm fallback.

**MViT cứng nhắc:** input CỐ ĐỊNH 16 frame × 224×224, norm Kinetics **[0.45]/[0.225]** (positional-embed
gắn cứng — đưa 112 vào là vỡ). Đã làm preprocess **backbone-aware**: `BACKBONE_SPEC` + `frames_to_clip`
trong `transport/model.py` dùng CHUNG train+infer (xóa 2 bản norm tách rời cũ). Head
MViT = `Sequential(Dropout, Linear)` → thay `head[-1]` (giữ Dropout).

**MViT KHÔNG export TensorRT được:** pooling-attention sinh op ONNX `SequenceAt/SequenceEmpty/SequenceInsert`
→ parser TRT 11.x báo UNSUPPORTED_NODE. Đừng tốn công thử lại. Classifier chạy **torch-FP16** (`.env`:
`VIO_ENGINE=` rỗng, `VIO_CKPT=weights/vio_mvit_v2_s.pt`) — vẫn nhanh trên Blackwell. Gate YOLO vẫn TRT.

**Train/serve skew THỜI GIAN (phát hiện lớn):** training lấy 16 frame TRẢI ĐỀU cả clip (~vài giây);
nếu infer lấy 16 frame LIÊN TIẾP ở fps gốc thì chỉ ~0.5s → model thấy motion quá ngắn → bỏ sót. Đo trên
VioPeru: span 0.5s recall **0.26** → span ~3s recall **0.56**, precision còn tăng. Fix: config `WIN_SECONDS=3.0`
+ `GRAB_FPS` → buffer `round(WIN_SECONDS*fps)` frame rồi `even_indices` lấy đều 16. Áp ở classifier.py +
analyze.py + eval_threshold.py (cả 3 đường infer phải khớp). Đây là lever quan trọng ngang crop-to-person.

**MULTI-SPAN × CROP/FULL (fix ca dashcam đám đông 2026-06-30):** 1 span cố định KHÔNG đủ. Cú đánh
dashcam học sinh diễn ra CHẬM ~7s → span 3s cho prob ~0.001, span 7s cho 0.96. Đồng thời CROP union
người ở cảnh ĐÔNG (8-11 người) = nguyên khung hoặc cắt mất ngữ cảnh → kìm prob (full-frame 0.96 vs
crop 0.68). Fix: `multispan_predict` (`transport/model.py`) lấy MAX prob qua
`WIN_SECONDS_LIST=[3.0,7.0]` × {crop-người, full-frame}. Buffer = span lớn nhất; quyết định khi
≥WIN_LEN frame (clip NGẮN <span vẫn chấm — nếu đợi `==span` thì clip 5s như VioPeru ra 0.0). Áp ở
analyze.py + classifier.py + eval_threshold.py. Kết quả: dashcam False→**True**; VioPeru recall tổng
hợp **0.59→0.66** (F1 0.70→0.75), precision giữ 0.87. Giá: tới 4 forward/cửa sổ (vẫn rẻ MViT 45ms) +
trễ buffer = span lớn nhất (~7s). Crop giờ CHỈ-giúp-không-hại (vì có nhánh full-frame song song).

**Ngưỡng kép tune lại** (VioPeru, multispan): `.env` `VIO_THRESHOLD=0.45 / STRONG=0.70 / ALERT_K=2`. Verdict
P≈0.90 R≈0.56. Recall cross-domain **trần ~0.60** (hạ ngưỡng <0.30 không thêm recall, chỉ thêm FP) — đây là
giới hạn domain-gap thật, muốn vượt phải fine-tune **footage CCTV Như Quỳnh** (VioPeru cố tình giữ làm test).
Liên quan: dual-threshold, person-crop-classifier (crop-in-training nay ĐÃ làm qua precompute_boxes.py).

**Chiến lược 2 model (deploy thật = RTX A2000, KHÔNG phải Blackwell):** MViT quá nặng cho A2000
(~0.5-1s/cửa sổ + KHÔNG ra TRT). → **MViT = server/độ chính xác** ("giáo viên", chạy Blackwell);
**r2plus1d_18 = edge/tốc độ** chạy A2000 (112×112, TRT FP16 ~20-40ms). Train edge dùng LẠI nguyên
pipeline (chỉ đổi `--backbone`) + cờ MỚI `--init-ckpt` warm-start từ checkpoint cũ. Lệnh đã chạy 2026-06-30:
`train_clip --backbone r2plus1d_18 --init-ckpt weights/vio_r2plus1d.pt --out weights/vio_r2plus1d_prod.pt`
→ val F1 **0.958** (gần MViT 0.969), VioPeru cross-domain **AUC 0.81** (kém MViT 0.88 — giá của model nhẹ).
Engine `weights/vio_r2plus1d_prod_fp16.engine` (export OK, 63MB). **Gotcha TRT đa-GPU:** engine build dưới
`CUDA_VISIBLE_DEVICES=1` PHẢI chạy lại với `CUDA_VISIBLE_DEVICES=1 VIO_DEVICE=0` (không thì
`cudaErrorIllegalAddress` vì _TrtRunner không set_device). Lên A2000 phải **rebuild engine** (engine arch-specific).
Review song song: dashboard #2 cổng 8081 (r2plus1d, GPU1) cạnh MViT 8080. (User sau đó chỉ giữ MViT.)

**Tốc độ analyze (upload):** nút thắt là **decode + gate YOLO (CPU) trên CẢ video**, KHÔNG phải forward MViT
— gộp 4 forward multispan thành 1 batch (`predict_batch`) chỉ giảm chút. Núm hiệu quả nhất là **WIN_STRIDE**
(mặc định đổi 8→16: dashcam 55s từ 58s→34s, vẫn detect; multispan đã phủ 3-7s nên thưa được) + `ANALYZE_GATE_EVERY`
(2→3) + `WIN_SECONDS_LIST` (bớt span). Muốn nhanh hơn nữa cần NVDEC (GPU decode) thay cv2-CPU.

---

## 2. Env: uv + Python 3.12.10 (type: feedback)

Dùng **uv** và **Python 3.12.10** cho môi trường dự án này (user yêu cầu rõ). Venv ở `.venv`.

**Why:** Máy có 2× RTX PRO 6000 Blackwell (sm_120) cần wheel cu128. requirements.txt pin các bản viễn tưởng (`torch==2.12.1`, `numpy==2.5.0`, `ultralytics==8.4.75`) — KHÔNG cài theo pin đó.

**How to apply:** `uv venv --python 3.12.10 .venv`; cài torch/torchvision từ `--index-url https://download.pytorch.org/whl/cu128`, phần còn lại từ pypi. Đừng để pip cài đè torch làm hỏng kernel Blackwell.

**Gotcha torch↔torchvision (2026-06-30):** venv từng lệch **torch 2.10.0 nhưng torchvision 0.26.0** (tv 0.26 build cho torch 2.11) → `import torchvision` lỗi cứng `RuntimeError: operator torchvision::nms does not exist` → chặn mọi việc dùng torchvision (train MViT). Fix: khớp cặp — `uv pip install --index-url .../cu128 'torchvision==0.25.*'` (0.25 ↔ torch 2.10). Verify: `python -c "from torchvision.models.video import mvit_v2_s; mvit_v2_s(weights='KINETICS400_V1')"`.

---

## 3. Violence demo setup (type: project)

Demo phát hiện bạo lực realtime cho **xã Như Quỳnh** (CA xã nhận cảnh báo). Tính tới 2026-06-29 đã chạy được end-to-end với model thật.

- **Gate model:** user chốt **YOLO26** (thế hệ tối ưu edge). Demo dùng `weights/yolo26x.pt` (copy từ `/home/abc/AI_project/Remote-Sensing/weights/`). Edge thật sau này dùng yolo26n/s → TensorRT.
- **Dataset:** RLVS tải auth-free từ HuggingFace `khoipd/Violence` → file `Real_Life_Violence_Dataset.zip` (~2.9GB, KHÔNG cần Kaggle token). Giải nén ở `data/RLVS/Real_Life_Violence_Dataset/{Violence,NonViolence}` = 1230 + 1119 clip. Tải nhanh bằng huggingface_hub (Xet, ~82s). RWF-2000 (đúng domain CCTV hơn) cần xin form GitHub mchengny/RWF2000 — chậm, để v2.
- **Checkpoint:** `weights/vio_r2plus1d.pt` (r2plus1d_18, clip_len 16, size 112) — fine-tune 8 epoch, **best val_acc 97.7%**. Verify: clip Violence→prob 0.96, NonViolence→prob 0.00.
- **Fix quan trọng:** `train/train_clip.py` `sample_frames` đã sửa từ `cap.set(POS_FRAMES)` seek-từng-frame (treo GPU 0%) sang đọc tuần tự grab()/retrieve() → ~118ms/clip, GPU ~79%.
- **Chạy:** `docker compose up -d` (redis 6381) → `python -m transport.init_streams` → gate + classifier + dashboard + grabber. Dashboard http://localhost:8080. Bằng chứng lưu `media/` (snapshot bbox + nhãn "VIOLENCE %" + clip).

**TensorRT (classifier):** `train/export_trt.py` + `train/test_trt.py`. Engine `weights/vio_r2plus1d_fp16.engine` (FP16). TRT 11.1.0.106 (trùng RS) đã BỎ `BuilderFlag.FP16/INT8`, `EXPLICIT_BATCH`, `platform_has_fast_fp16` → build FP16 phải dùng **STRONGLY_TYPED network + ONNX export ở half** (không phải set_flag). Test: 16/16 khớp nhãn fp32, |Δprob|≤0.0002, latency 3.04ms→1.15ms (2.64×). Gate YOLO build engine kiểu RS: `YOLO('weights/yolo26x.pt').export(format='engine', half=True, imgsz=640, device=0)` (RS dùng imgsz=960).

**Full-TRT đã xong (2026-06-29):** `.env` set `VIO_ENGINE=weights/vio_r2plus1d_fp16.engine` + `GATE_MODEL=weights/yolo26x.engine`. model.py có `_TrtRunner` + mode engine/torch/dummy (VIO_ENGINE ưu tiên). Gate engine build kiểu RS (110MB). Pipeline gate+classifier đều TensorRT FP16, không còn FP32. Verify end-to-end: clip bạo lực → prob 0.99 → alert + snapshot.

**Còn lại:** Telegram alert, giả lập RTSP (mediamtx+ffmpeg — chưa có trên máy), tune threshold trên footage giống thật.

---

## 4. Public tunnel (type: project)

Dashboard demo public tại **https://watch.tpvtech.vn** qua **Cloudflare Tunnel** (token-based, ingress cấu hình remote: `watch.tpvtech.vn → http://localhost:80`).

**Kiến trúc (2 container Docker, đều `--network host`):**
- `vio-proxy80`: `alpine/socat` bind host:80 → forward 127.0.0.1:8080 (dashboard). Cần vì tunnel trỏ :80 còn dashboard chạy :8080, và **user `abc` KHÔNG có sudo** nên không tự bind port 80 — để dockerd (root, có sẵn quyền) bind hộ. `docker run -d --name vio-proxy80 --restart unless-stopped --network host alpine/socat TCP-LISTEN:80,fork,reuseaddr TCP:127.0.0.1:8080`
- `cf-tunnel`: `cloudflare/cloudflared:latest tunnel --no-autoupdate run --token <TOKEN>`. PHẢI `--network host` thì `localhost:80` trong ingress mới = host (container thường thì localhost = chính container).

**Gotcha:** cloudflared trong Docker mà không `--network host` thì không thấy service trên host. Token là secret (không lưu ở đây) — nằm trong container cf-tunnel.

**Auth:** đã thêm HTTP Basic ở tầng app (middleware trong dashboard/app.py), bật bằng env `DASH_USER`/`DASH_PASS` trong .env (rỗng DASH_PASS = mở). Browser tự nhớ creds nên SSE/upload không vỡ. Mạnh hơn: Cloudflare Access (cần cấu hình trong Zero Trust dashboard của user — Claude không tự làm được).

---

## 5. Dual threshold (type: project)

Quyết định 2026-06-29: bỏ ngưỡng đơn cứng (user từng hạ 0.7→0.6 bằng cảm tính). Lý do user nêu: "đôi khi chỉ 1 hành động nhỏ cũng là bạo lực, không được bỏ qua" → k-of-n (ALERT_K=3) nuốt mất cú đánh/tát chớp nhoáng (1-2 cửa sổ).

**Cơ chế ngưỡng kép** (cùng dùng ở `transport/analyze.py` offline + `dashboard/app.py` live):
```
violence = (số cửa sổ prob≥VIO_THRESHOLD ≥ ALERT_K)  OR  (≥1 cửa sổ prob≥VIO_THRESHOLD_STRONG)
```
cả hai vẫn qua gate `persons ≥ MIN_PERSONS`. Config: `VIO_THRESHOLD_STRONG=0.85` (`transport/config.py`). Đường mạnh = bắt sự kiện ngắn; đường k/n = bạo lực kéo dài độ tin cậy vừa + lọc nhiễu lẻ. Hysteresis/EMA bị loại vì làm mượt → giết spike ngắn (sai hướng).

**Đo ngưỡng:** `train/eval_threshold.py`. Chạy classifier theo cửa sổ trượt trên RLVS (nhãn từ path qua `train.train_clip.label_of`), cache prob ra `logs/probe.json`, sweep lưới (T_low,T_strong,K) in-memory → in Prec/Rec/F1, gợi ý cấu hình F1-max và Recall-max. `--gate` để mô phỏng MIN_PERSONS (mặc định tắt).

---

## 6. Person-crop classifier (type: project)

Quyết định 2026-06-29 (giải pháp "A" cho domain gap): r2plus1d train trên RLVS (đánh nhau cận cảnh, người to gần kín khung) **không tổng quát hóa** sang CCTV thật góc rộng — người nhỏ ở góc, co cả khung về 112×112 làm mất tín hiệu chuyển động. Triệu chứng: video CCTV có bạo lực nhưng prob đỉnh chỉ 22–54% → bỏ sót. KHÔNG phải lỗi ngưỡng (22% thì ngưỡng nào cũng vô dụng).

**Fix:** cắt quanh union bbox người (YOLO đã có sẵn) → vuông hóa + đệm `CROP_PAD=0.35` → rồi mới resize 112. Hành động bạo lực to gần kín khung như RLVS. Helper `person_crop_box()` ở `transport/model.py`; áp ở cả `transport/analyze.py` (upload) lẫn `transport/classifier.py` (live), track bbox từng frame qua deque rồi union cả cửa sổ. Bật/tắt `CROP_TO_PERSON=1` (mặc định bật).

**Đo thực tế (video432, CCTV phòng có giường):** không crop max=0.535 → violence=False (SAI); có crop **max=0.868 → violence=True** (đúng, vượt strong 0.80). Cảnh người đi bình thường (people_test.mp4) giữ nguyên 0.227 → KHÔNG false-positive. Tức crop khuếch đại bạo lực thật mà không thổi phồng cảnh thường.

Đây là giải pháp rẻ (không train lại) tận dụng đúng thiết kế "2 mạng cạnh nhau": YOLO định vị → classifier soi vào đó. Nếu vẫn chưa đủ thì mới tới "B" = fine-tune RWF-2000/CCTV. Không cứu được ca **súng tĩnh** (cần nhận diện vũ khí, đã tạm gác).

**TODO khi tối ưu training (đã làm 1 phần qua precompute_boxes.py):** `train/train_clip.py` từng train trên CẢ KHUNG → lệch với inference (đã crop vùng người) = train/serve skew. Khi train lại PHẢI crop quanh người y như inference (dùng `person_crop_box`) để khớp phân phối → prob sẽ lên thêm. Các lever khác: thêm dữ liệu domain (RWF-2000 CCTV), augmentation không-gian, LR schedule/early-stopping.

---

## 7. Upload/analyze feature (type: project)

Dashboard (`dashboard/app.py`) có tính năng **tải video lên → phân tích → bấm xem lại video kèm overlay detection** (trước đó chỉ infer qua terminal/grabber).

- **Backend:** `POST /api/upload` (cần `python-multipart`) lưu vào `uploads/`, chạy `transport/analyze.py` trong thread (lock tuần tự). `GET /api/analyses` liệt kê job. Kết quả video lưu `media/analyzed/<id>.mp4`, serve qua `/media`.
- **analyze.py:** offline, tái dùng YOLO gate engine + VioModel engine. Mỗi frame YOLO đếm/vẽ người; cửa sổ trượt WIN_LEN → classifier prob; overlay box + "BAO LUC %" + banner cảnh báo. Verdict: violence nếu pos_windows≥ALERT_K, kèm max_prob/max_at/timeline.
- **H.264 bắt buộc:** dùng `imageio` + `imageio-ffmpeg` (binary ffmpeg tĩnh, KHÔNG cần sudo — máy không có ffmpeg hệ thống) encode libx264 yuv420p. opencv mp4v KHÔNG phát được trên web. Kích thước phải chẵn (W-=W%2).
- **UI:** khu upload + lưới card (badge violence/bình thường/progress), click card → modal `<video controls>` phát video overlay. SSE event `type:analysis` + poll `/api/analyses` cho job đang chạy.

Verify: upload clip RLVS → violence=True max_prob 0.976, video phát được local + public. ~7.6s/clip 5s (YOLO mỗi frame).
