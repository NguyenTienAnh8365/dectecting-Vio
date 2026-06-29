#!/usr/bin/env bash
# Dựng cả pipeline demo bằng 1 clip → mở http://localhost:8080
# Dùng: scripts/run_demo.sh data/fight.mp4 [cam_id]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
CLIP="${1:?cần đường dẫn clip, vd: scripts/run_demo.sh data/fight.mp4}"
CAM="${2:-cam1}"

$PY -m transport.init_streams

mkdir -p logs
$PY -m transport.gate        > logs/gate.log 2>&1 &       echo "gate pid $!"
$PY -m transport.classifier  > logs/classifier.log 2>&1 & echo "classifier pid $!"
$PY -m dashboard.app         > logs/dashboard.log 2>&1 &  echo "dashboard pid $!"
sleep 4   # chờ model nạp xong

echo "→ Dashboard: http://localhost:8080"
echo "→ Đẩy clip vào (loop). Ctrl-C để dừng grabber; các tiến trình nền vẫn chạy."
$PY -m transport.grabber --src "$CLIP" --camera "$CAM" --fps 12 --loop

echo "Dừng nền: pkill -f 'transport\\.|dashboard\\.app'"
