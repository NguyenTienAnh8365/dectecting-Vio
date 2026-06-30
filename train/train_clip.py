"""Fine-tune classifier clip-level (violence) — bản PRODUCTION (đa-dataset, MViT-ready).

Khác bản demo cũ:
- Đọc từ `data/manifest.csv` (build_manifest.py) thay vì glob+heuristic → nhãn & split tin cậy.
- CROP-TO-PERSON trong training (đọc cache precompute_boxes.py) → KHỚP đúng cách infer crop,
  hết train/serve skew (README #1). Tắt bằng --no-crop.
- Preprocess DÙNG CHUNG `transport.model.frames_to_clip` (size/normalize theo BACKBONE_SPEC) →
  train↔infer đồng nhất tuyệt đối. mvit_v2_s tự lấy 224×224 + norm [0.45]/[0.225].
- Augmentation không-gian (random-resized-crop + color jitter + h-flip, nhất quán cả clip),
  optional mixup clip.
- AMP, AdamW param-groups (LR thấp cho backbone, cao cho head), warmup→cosine, class-weighted CE,
  EARLY STOPPING theo val-F1, metrics P/R/F1/AUC. Checkpoint lưu metadata đầy đủ để export+infer tái lập.

Dùng (MViT, full):
  uv run python -m train.train_clip --backbone mvit_v2_s --epochs 60 --batch 16 \
      --out weights/vio_mvit_v2_s.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transport.model import build_backbone, frames_to_clip, spec_of

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


# ----------------------------- data ----------------------------------------
def load_manifest(path: str, split: str) -> list[tuple[str, int]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append((os.path.join(ROOT, r["path"]), int(r["label"])))
    return rows


def sample_frames_bgr(path: str, clip_len: int, train: bool) -> list | None:
    """clip_len frame BGR trải đều (train: jitter). Đọc tuần tự grab/retrieve (nhanh với h264)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n > 0:
        idxs = np.linspace(0, n - 1, clip_len)
        if train:
            idxs = np.clip(idxs + np.random.randint(-2, 3, clip_len), 0, n - 1)
        want = idxs.astype(int)
        order = np.argsort(want, kind="stable")
        cur, by_pos, last = -1, {}, None
        for pos in order:
            target = int(want[pos])
            while cur < target:
                if not cap.grab():
                    break
                cur += 1
            ok, fr = cap.retrieve() if cur == target else (False, None)
            if ok:
                last = fr
            by_pos[pos] = fr if ok else last
        cap.release()
        sel = [by_pos.get(i) for i in range(clip_len)]
        valid = next((f for f in sel if f is not None), None)
        if valid is None:
            return None
        return [f if f is not None else valid for f in sel]
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    idxs = np.linspace(0, len(frames) - 1, clip_len).astype(int)
    return [frames[i] for i in idxs]


def _augment(frames: list, rng: random.Random) -> list:
    """Aug KHÔNG-GIAN nhất quán cả clip: random-resized-crop + color jitter (brightness/contrast).

    H-flip xử lý riêng qua frames_to_clip(flip=...). Giữ nhẹ để không phá tín hiệu CCTV.
    """
    h, w = frames[0].shape[:2]
    # random-resized-crop (cùng 1 rect cho mọi frame)
    if rng.random() < 0.8:
        area = h * w
        x0 = y0 = 0
        cw, ch = w, h
        for _ in range(10):
            ta = area * rng.uniform(0.7, 1.0)
            ar = rng.uniform(3 / 4, 4 / 3)
            cw_ = int(round(math.sqrt(ta * ar)))
            ch_ = int(round(math.sqrt(ta / ar)))
            if 0 < cw_ <= w and 0 < ch_ <= h:
                cw, ch = cw_, ch_
                x0 = rng.randint(0, w - cw)
                y0 = rng.randint(0, h - ch)
                break
        frames = [f[y0:y0 + ch, x0:x0 + cw] for f in frames]
    # color jitter (cùng hệ số cho mọi frame)
    if rng.random() < 0.7:
        b = rng.uniform(0.85, 1.15)   # brightness
        c = rng.uniform(0.85, 1.15)   # contrast quanh 128
        out = []
        for f in frames:
            g = f.astype(np.float32)
            g = (g - 128.0) * c + 128.0 * b
            out.append(np.clip(g, 0, 255).astype(np.uint8))
        frames = out
    return frames


class ClipDataset(Dataset):
    def __init__(self, rows, backbone, clip_len, train, crop_boxes=None, augment=True):
        self.rows = rows
        self.backbone = backbone
        self.clip_len = clip_len
        self.train = train
        self.crop_boxes = crop_boxes or {}
        self.augment = augment and train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, label = self.rows[i]
        frames = sample_frames_bgr(path, self.clip_len, self.train)
        if frames is None:
            return self.__getitem__((i + 1) % len(self.rows))
        # crop-to-person (từ cache) — KHỚP infer
        rel = os.path.relpath(path, ROOT)
        info = self.crop_boxes.get(rel)
        if info and info.get("box"):
            x1, y1, x2, y2 = info["box"]
            frames = [f[y1:y2, x1:x2] for f in frames]
        # augmentation
        flip = False
        if self.augment:
            rng = random.Random()
            frames = _augment(frames, rng)
            flip = rng.random() < 0.5
        arr = frames_to_clip(frames, self.backbone, flip=flip)  # [C,T,H,W]
        return torch.from_numpy(arr), label


# ----------------------------- metrics -------------------------------------
def binary_metrics(probs: list[float], labels: list[int]) -> dict:
    n = len(labels)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": _auc(probs, labels)}


def _auc(probs: list[float], labels: list[int]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # rank trung bình cho ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)


# ----------------------------- train ---------------------------------------
def build_optimizer(model, lr, lr_backbone_mult, wd):
    """2 nhóm: head (lr đầy đủ) + backbone (lr*mult thấp hơn) — chuẩn fine-tune transformer."""
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith(("head", "fc")) else backbone).append(p)
    return torch.optim.AdamW(
        [{"params": backbone, "lr": lr * lr_backbone_mult},
         {"params": head, "lr": lr}], lr=lr, weight_decay=wd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--backbone", default="mvit_v2_s")
    ap.add_argument("--epochs", type=int, default=60, help="trần epoch (early-stop có thể dừng sớm)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8, help="early-stop nếu val-F1 không cải thiện")
    ap.add_argument("--clip-len", type=int, default=0, help="0=lấy theo backbone spec")
    ap.add_argument("--crop-boxes", default=os.path.join(ROOT, "data", "cache", "crop_boxes.json"))
    ap.add_argument("--no-crop", action="store_true", help="tắt crop-to-person (train cả khung)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0, help="alpha mixup clip (0=tắt)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-balance", action="store_true", help="tắt class-weighted loss")
    ap.add_argument("--init-ckpt", default="", help="warm-start: nạp weights từ .pt này thay vì chỉ Kinetics")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="weights/vio_mvit_v2_s.pt")
    args = ap.parse_args()

    sp = spec_of(args.backbone)
    clip_len = args.clip_len or sp["clip_len"]
    size = sp["size"]
    dev = args.device
    amp = not args.no_amp and dev != "cpu"

    crop_boxes = {}
    if not args.no_crop:
        if os.path.exists(args.crop_boxes):
            import json
            crop_boxes = json.load(open(args.crop_boxes, encoding="utf-8"))
            n_box = sum(1 for v in crop_boxes.values() if v.get("box"))
            print(f"[crop] cache {len(crop_boxes)} clip ({n_box} có hộp người) ← {args.crop_boxes}")
        else:
            print(f"[crop] ⚠ KHÔNG thấy {args.crop_boxes} → train cả khung (chạy precompute_boxes trước!)")

    tr_rows = load_manifest(args.manifest, "train")
    va_rows = load_manifest(args.manifest, "val")
    pos = sum(l for _, l in tr_rows)
    print(f"train={len(tr_rows)} (pos={pos} neg={len(tr_rows)-pos})  val={len(va_rows)}  "
          f"backbone={args.backbone} size={size} clip_len={clip_len} amp={amp} crop={not args.no_crop}")

    tr = DataLoader(ClipDataset(tr_rows, args.backbone, clip_len, True, crop_boxes, not args.no_augment),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    va = DataLoader(ClipDataset(va_rows, args.backbone, clip_len, False, crop_boxes, False),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Warm-start: nạp checkpoint cũ (vd r2plus1d 8-epoch RLVS) → kế thừa cái đã học, hội tụ nhanh hơn.
    model = build_backbone(args.backbone, num_classes=2, pretrained=not args.init_ckpt)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu")
        miss = model.load_state_dict(ck.get("model", ck), strict=False)
        print(f"[init] warm-start từ {args.init_ckpt} "
              f"(val_acc cũ={ck.get('val_acc')}, missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)})")
    model = model.to(dev)
    opt = build_optimizer(model, args.lr, args.lr_backbone_mult, args.wd)

    # warmup tuyến tính → cosine
    def lr_factor(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    weight = None
    if not args.no_balance:
        neg = len(tr_rows) - pos
        w = torch.tensor([len(tr_rows) / (2 * neg), len(tr_rows) / (2 * pos)], dtype=torch.float32)
        weight = w.to(dev)
        print(f"[balance] class weights = {w.tolist()}")
    crit = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifest_hash = hashlib.sha1(open(args.manifest, "rb").read()).hexdigest()[:12]
    best_f1, best_ep, since = -1.0, -1, 0

    for ep in range(args.epochs):
        model.train()
        run, tot = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                if args.mixup > 0:
                    lam = np.random.beta(args.mixup, args.mixup)
                    perm = torch.randperm(x.size(0), device=dev)
                    x = lam * x + (1 - lam) * x[perm]
                    loss = lam * crit(model(x), y) + (1 - lam) * crit(model(x), y[perm])
                else:
                    loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            tot += x.size(0)
        sched.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in va:
                x = x.to(dev, non_blocking=True)
                with torch.autocast("cuda", enabled=amp):
                    p = torch.softmax(model(x).float(), dim=1)[:, 1]
                probs += p.detach().cpu().tolist()
                labels += y.tolist()
        m = binary_metrics(probs, labels)
        lr_now = sched.get_last_lr()[0]
        print(f"epoch {ep+1}/{args.epochs} loss={run/max(tot,1):.4f} lr={lr_now:.2e} "
              f"val: acc={m['acc']:.4f} P={m['prec']:.4f} R={m['rec']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}",
              flush=True)

        if m["f1"] > best_f1:
            best_f1, best_ep, since = m["f1"], ep + 1, 0
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "clip_len": clip_len, "size": size, "mean": sp["mean"], "std": sp["std"],
                        "crop_to_person": not args.no_crop, "val": m, "epoch": ep + 1,
                        "manifest_sha1": manifest_hash}, args.out)
            print(f"  ✓ saved {args.out} (val_F1={best_f1:.4f})", flush=True)
        else:
            since += 1
            if since >= args.patience:
                print(f"early-stop: val-F1 không cải thiện {args.patience} epoch (best={best_f1:.4f} @ep{best_ep})")
                break

    print(f"done. best val_F1={best_f1:.4f} @epoch{best_ep} → {args.out}")


if __name__ == "__main__":
    main()
