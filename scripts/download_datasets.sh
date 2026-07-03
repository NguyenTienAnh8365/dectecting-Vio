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

echo "== UBI-Fights (mirror Kaggle — trang gốc socia-lab chết; kagglehub KHÔNG cần API key) =="
$PY - <<'PY'
import kagglehub
print("→", kagglehub.dataset_download("intissarziani/ubi-fightsall"))
PY
$PY scripts/build_ubifights_clips.py

echo "== UCF-Crime (HF jinmang2/ucf_crime — chỉ Part-1 Abuse/Assault + Part-2 Fighting + Normal) =="
$PY - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["Anomaly-Videos-Part-1.zip","Anomaly-Videos-Part-2.zip",
          "Testing_Normal_Videos.zip","Normal_Videos_for_Event_Recognition.zip"]:
    hf_hub_download("jinmang2/ucf_crime", f, repo_type="dataset", local_dir="data/_dl/ucfcrime")
# annotation temporal phủ CẢ train (chính thức CRCV chỉ có test)
for f in ["ucf_database_train_step13_5sec_3sec_category.json",
          "ucf_database_test_step13_5sec_3sec_category.json"]:
    hf_hub_download("backseollgi/UCF_Crime_ICML_annotation", f, repo_type="dataset", local_dir="data/_dl/ucfcrime")
PY
mkdir -p data/_dl/ucfcrime/raw
unzip -q -o data/_dl/ucfcrime/Anomaly-Videos-Part-1.zip 'Anomaly-Videos-Part-1/Abuse/*' 'Anomaly-Videos-Part-1/Assault/*' -d data/_dl/ucfcrime/raw
unzip -q -o data/_dl/ucfcrime/Anomaly-Videos-Part-2.zip 'Anomaly-Videos-Part-2/Fighting/*' -d data/_dl/ucfcrime/raw
unzip -q -o data/_dl/ucfcrime/Testing_Normal_Videos.zip -d data/_dl/ucfcrime/raw
unzip -q -o data/_dl/ucfcrime/Normal_Videos_for_Event_Recognition.zip -d data/_dl/ucfcrime/raw
$PY scripts/build_ucfcrime_clips.py

echo "DONE. Tiếp: $PY -m train.build_manifest  (data/exclude_clips.txt tự được áp nếu có)"
