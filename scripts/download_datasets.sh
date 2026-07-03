#!/usr/bin/env bash
# Tải 4 dataset bạo lực (auth-free) về data/. RLVS tải riêng (HF khoipd/Violence) — xem memory.
# Dùng: scripts/download_datasets.sh        (chạy từ gốc repo, cần .venv + git)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "== RWF-2000 (HF DanJoshua/RWF-2000, ~12GB tar.gz) =="
$PY - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("DanJoshua/RWF-2000","RWF-2000.tar.gz",repo_type="dataset",local_dir="data/_dl")
PY
mkdir -p data/RWF2000
tar xzf data/_dl/RWF-2000.tar.gz -C data/RWF2000 --strip-components=1 || true  # vài tên file dài lỗi → bỏ qua

echo "== VioPeru (GitHub, 280 clip CCTV Peru — TEST set) =="
[ -d data/VioPeru ] || git clone --depth 1 https://github.com/hhuillcen/VioPeru.git data/VioPeru

echo "== Surveillance Fight (GitHub, 300 clip) =="
[ -d data/SurveillanceFight ] || git clone --depth 1 https://github.com/sayibet/fight-detection-surv-dataset.git data/SurveillanceFight

echo "== SCVD (HF parquet-frame → tái dựng mp4, đã loại Weaponized) =="
$PY scripts/build_scvd_clips.py

echo "DONE. Tiếp: $PY -m train.build_manifest"
# UBI-Fights + UCF-Crime từng thêm rồi GỠ (không giảm gap camera lạ — xem README
# "KẾT QUẢ đợt mở rộng"). Script tải/cắt nằm ở git history (commit da9d5e0) nếu cần lại.
