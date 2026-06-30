"""Precompute crop-box người cho TỪNG clip trong manifest → cache JSON (cho crop-in-training).

Lý do (README #1): infer đã crop quanh người TRƯỚC classifier, nhưng training cũ train CẢ KHUNG →
train/serve skew. Ở đây chạy YOLO gate (yolo26x.engine) MỘT LẦN trên ~N_PROBE frame rải đều mỗi
clip, union mọi bbox người qua `person_crop_box` (vuông + đệm CROP_PAD) → 1 hộp crop/clip, đem
cache. Dataset lúc train chỉ việc crop theo hộp này → thấy hành động "to gần kín khung" y như infer,
KHÔNG phải chạy YOLO mỗi epoch.

Chạy:  uv run python -m train.precompute_boxes              # 1 GPU
       VIO/GATE chạy 2 GPU: --shards 2 --shard 0 (device 0) & --shards 2 --shard 1 (device 1)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np

from transport import config
from transport.model import person_crop_box, quiet_ml_logs

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
N_PROBE = 24  # số frame rải đều/clip để dò người (đủ để union, rẻ)


def _persons(res) -> list[list[int]]:
    """bbox [x,y,w,h] góc trên-trái của 'person' từ 1 kết quả ultralytics."""
    out = []
    b = res.boxes
    if b is None:
        return out
    xywh = b.xywh.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    for (x, y, w, h), c in zip(xywh, cls):
        if int(c) == config.PERSON_CLASS:
            out.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
    return out


def _probe_frames(path: str):
    """Đọc N_PROBE frame rải đều (tuần tự grab/retrieve) → (frames, W, H) hoặc (None,...)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            return None, 0, 0
        idxs = np.linspace(0, len(frames) - 1, min(N_PROBE, len(frames))).astype(int)
        sel = [frames[i] for i in idxs]
        H, W = sel[0].shape[:2]
        return sel, W, H
    want = sorted(set(np.linspace(0, n - 1, min(N_PROBE, n)).astype(int).tolist()))
    sel, cur, wi = [], -1, 0
    while wi < len(want):
        if not cap.grab():
            break
        cur += 1
        if cur == want[wi]:
            ok, fr = cap.retrieve()
            if ok:
                sel.append(fr)
            wi += 1
    cap.release()
    if not sel:
        return None, W, H
    if not W or not H:
        H, W = sel[0].shape[:2]
    return sel, W, H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "crop_boxes.json"))
    ap.add_argument("--device", default=config.GATE_DEVICE)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = [r["path"] for r in csv.DictReader(f)]
    rows = [p for i, p in enumerate(rows) if i % args.shards == args.shard]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_path = args.out if args.shards == 1 else args.out + f".{args.shard}"
    cache: dict = {}
    if os.path.exists(out_path):
        cache = json.load(open(out_path, encoding="utf-8"))

    quiet_ml_logs()
    from ultralytics import YOLO
    gate = YOLO(config.GATE_MODEL, task="detect", verbose=False)

    done = 0
    for rel in rows:
        if rel in cache:
            continue
        path = os.path.join(ROOT, rel)
        frames, W, H = _probe_frames(path)
        if not frames:
            cache[rel] = {"box": None, "w": W, "h": H, "bad": True}
        else:
            # Engine YOLO build batch tĩnh =1 → predict TỪNG frame (như analyze.py).
            boxes_per_frame = [
                _persons(gate.predict(fr, imgsz=config.GATE_IMGSZ, conf=config.GATE_CONF,
                                      classes=[config.PERSON_CLASS], quantize=config.GATE_QUANTIZE,
                                      device=args.device, verbose=False)[0])
                for fr in frames
            ]
            box = person_crop_box(boxes_per_frame, W, H, config.CROP_PAD)
            cache[rel] = {"box": list(box) if box else None, "w": W, "h": H}
        done += 1
        if done % 200 == 0:
            json.dump(cache, open(out_path, "w"), separators=(",", ":"))
            print(f"  {done} clip... (vừa: {rel})")

    json.dump(cache, open(out_path, "w"), separators=(",", ":"))
    n_box = sum(1 for v in cache.values() if v.get("box"))
    print(f"done. cache {len(cache)} clip → {out_path}  (có người/crop: {n_box}, "
          f"không người: {len(cache)-n_box})")


if __name__ == "__main__":
    main()
