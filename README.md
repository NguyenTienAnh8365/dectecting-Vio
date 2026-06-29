# Phát hiện bạo lực realtime từ camera — demo xã Như Quỳnh

Phát hiện hành vi **gây gổ / xô đẩy / đánh nhau** từ camera, **sáng đèn dashboard +
snapshot/clip** cho CA xã xác nhận và có mặt kịp thời. Human-in-the-loop: hệ thống là
**decision-support**, người trực gật đầu cuối cùng — KHÔNG tự động điều người.

Tái dùng tầng transport (Redis Streams + YOLO + reliability) của dự án **Remote Sensing**.
Khác biệt cốt lõi: bài toán này **temporal (clip-level)**, không per-frame → thêm 1 stage
**stateful per-camera** (cửa sổ trượt + 3D-CNN).

```
RTSP/clip ─grabber─> [vio:frames:ready] ─gate (YOLO person, stateless, group)─>
  [vio:frames:gated] ─classifier (STATEFUL per-camera, cửa sổ 16f, 3D-CNN)─>
    [vio:results] ─dashboard (k-of-n + cooldown)─> 🔔 SSE + snapshot/clip (disk)
```

| Stage | File | Tái dùng từ RS | Vai trò |
|---|---|---|---|
| grabber | [transport/grabber.py](transport/grabber.py) | producer.py | clip/RTSP → frame stream |
| person-gate | [transport/gate.py](transport/gate.py) | worker.py | YOLO đếm người + bbox (stateless, group) |
| classifier | [transport/classifier.py](transport/classifier.py) | 🆕 mới | cửa sổ trượt per-camera → prob bạo lực |
| model | [transport/model.py](transport/model.py) | 🆕 mới | torchvision video (r2plus1d_18…) |
| dashboard | [dashboard/app.py](dashboard/app.py) | consumer.py | k-of-n + cooldown → alert + SSE |
| store | [transport/store.py](transport/store.py) | storage.py (MinIO→disk) | snapshot + clip bằng chứng |

## Cài đặt (server Blackwell)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # sửa REDIS_PASSWORD nếu cần
docker compose up -d          # Redis ở cổng 6381
python -m transport.init_streams
```

## 1) Lấy data & fine-tune (Ngày 0–1)

- **RLVS** (Real Life Violence Situations, ~2000 clip) — tải ngay từ Kaggle, không cần duyệt → bắt đầu luôn.
- **RWF-2000** (CCTV, đúng domain giám sát nhất) — xin quyền form GitHub `mchengny/RWF2000` **ngay hôm nay** vì duyệt mất thời gian.

```bash
python -m train.train_clip --data data/RLVS --backbone r2plus1d_18 \
    --epochs 8 --batch 8 --out weights/vio_r2plus1d.pt
# rồi trỏ checkpoint trong .env:  VIO_CKPT=weights/vio_r2plus1d.pt
```

> Chưa có checkpoint? Pipeline vẫn chạy thông ở **dummy mode** (prob=0, không bắn alert) để
> test transport/dashboard trước khi train xong.

## 2) Chạy demo end-to-end

```bash
scripts/run_demo.sh data/fight_sample.mp4 cam1     # dựng đủ 4 tiến trình + đẩy clip (loop)
# mở http://<server>:8080
```

Hoặc chạy tay từng tiến trình (mỗi cái 1 terminal):
```bash
python -m transport.gate
python -m transport.classifier
python -m dashboard.app
python -m transport.grabber --src data/fight_sample.mp4 --camera cam1 --fps 12 --loop
```

**Camera thật** chỉ đổi `--src`, không sửa code:
```bash
python -m transport.grabber --src "rtsp://user:pass@ip:554/Streaming/Channels/101" --camera cam1 --fps 12
```

**Giả lập RTSP từ clip** (chứng minh đường RTSP thật) — dùng mediamtx + ffmpeg loop, rồi trỏ grabber vào `rtsp://localhost:8554/cam1`.

## Tinh chỉnh chống false-positive ([transport/config.py](transport/config.py))

| Nhóm | Biến | Mặc định | Ghi chú |
|---|---|---|---|
| Cửa sổ | `WIN_LEN` · `WIN_STRIDE` | 16 · 8 | clip 16 frame, phân loại lại mỗi 8 frame |
| Ngưỡng | `VIO_THRESHOLD` | 0.70 | prob 1 cửa sổ tính là "dương" |
| Smoothing | `ALERT_K` / `ALERT_N` | 3 / 5 | cần 3/5 cửa sổ dương → alert (chống nhiễu) |
| Spam | `COOLDOWN_S` | 45 | mỗi camera tối đa 1 alert / 45s |
| Person-gate | `MIN_PERSONS` | 2 | <2 người thì không alert (loại cảnh 1 người) |
| Lưu | `SAVE_PROB` | 0.50 | prob ≥ mức này mới lưu snapshot+clip |

Tune theo footage thật ở Ngày 3: trẻ con đùa/ôm nhau/đám đông là nguồn false-positive chính → nâng `VIO_THRESHOLD`/`ALERT_K` nếu báo nhầm nhiều.

## Timeline 2–3 ngày
- **Ngày 0**: xin quyền RWF-2000; tải RLVS; `docker compose up`, chạy dummy end-to-end cho thông.
- **Ngày 1**: fine-tune r2plus1d_18 trên RLVS; validate trên clip test.
- **Ngày 2**: ráp full pipeline + dashboard, tune ngưỡng/k-of-n/cooldown.
- **Ngày 3**: test footage giống thật, chống false-positive, polish demo.

## Pháp lý
Giám sát công + xử lý hình ảnh người dân → **Nghị định 13/2023/NĐ-CP** áp dụng. Đã có:
TTL tự xóa media (`MEDIA_TTL_DAYS`, mặc định 3 ngày) + human-in-the-loop. Cần bổ sung cho
production: phân quyền truy cập dashboard, policy lưu trữ/xóa, nhật ký truy cập.

## Đường lên Jetson (sau demo)
Stage **gate** và **classifier** stateless về contract (chỉ giao tiếp qua Redis) → bê thẳng
lên Jetson: export model TensorRT (`yolo … format=engine`; classifier → ONNX→TRT), giữ
nguyên stream/dashboard. Server Blackwell làm "lò luyện" fine-tune + distill.
