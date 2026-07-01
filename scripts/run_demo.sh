#!/usr/bin/env bash
# Dựng pipeline demo bằng 1 clip → mở http://localhost:8080
# Camera nối trực tiếp vào dashboard.app (transport/live.py, không qua Redis/tiến trình rời).
# Dùng: scripts/run_demo.sh data/fight.mp4 [cam_id]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
CLIP="${1:?cần đường dẫn clip, vd: scripts/run_demo.sh data/fight.mp4}"
CAM="${2:-cam1}"

mkdir -p logs
LIVE_CAMS="${CAM}=${CLIP}" LIVE_LOOP=1 $PY -m dashboard.app > logs/dashboard.log 2>&1 &
echo "dashboard pid $!"

echo "→ Dashboard: http://localhost:8080 (camera '$CAM' đọc lặp '$CLIP')"
echo "Dừng: pkill -f 'dashboard\\.app'"
