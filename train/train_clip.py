"""Fine-tune classifier clip-level (violence) trên RLVS / RWF-2000.

Backbone Kinetics-pretrained (torchvision) → đầu nhị phân. Trên 1 Blackwell chỉ vài giờ.

Cấu trúc data: thư mục chứa video .mp4/.avi, nhãn suy từ ĐƯỜNG DẪN:
  - dương (violence/fight) : path chứa 'fight' hoặc 'viol' (và KHÔNG chứa 'non'/'normal')
  - âm (non-violence)      : còn lại
Hợp cả RLVS (Violence/ NonViolence/) lẫn RWF-2000 (train|val / Fight|NonFight).

Dùng:
  # RLVS tải từ Kaggle, RWF-2000 xin quyền GitHub mchengny/RWF2000
  python -m train.train_clip --data data/RLVS --epochs 8 --batch 8 --out weights/vio_r2plus1d.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transport.model import build_backbone

_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mpg", "*.mpeg")


def label_of(path: str) -> int:
    p = path.lower()
    neg = ("non" in p) or ("normal" in p)
    pos = ("fight" in p) or ("viol" in p)
    return 1 if (pos and not neg) else 0


def sample_frames(path: str, clip_len: int, size: int, train: bool) -> np.ndarray | None:
    """Lấy clip_len frame trải đều (train: jitter ngẫu nhiên) → [T,H,W,C] RGB chuẩn hóa."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
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
        sel = [frames[i] for i in idxs]
    else:
        idxs = np.linspace(0, n - 1, clip_len)
        if train:
            idxs = np.clip(idxs + np.random.randint(-2, 3, clip_len), 0, n - 1)
        sel = []
        for i in idxs.astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                ok, fr = cap.read()
            sel.append(fr if ok else (sel[-1] if sel else np.zeros((size, size, 3), np.uint8)))
        cap.release()

    out = []
    flip = train and random.random() < 0.5
    for fr in sel:
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        if flip:
            fr = cv2.flip(fr, 1)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append((fr - _MEAN) / _STD)
    return np.stack(out, 0)  # [T,H,W,C]


class ClipDataset(Dataset):
    def __init__(self, files: list[str], clip_len: int, size: int, train: bool):
        self.files = files
        self.clip_len = clip_len
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        arr = sample_frames(path, self.clip_len, self.size, self.train)
        if arr is None:  # video hỏng → thử cái khác
            return self.__getitem__((i + 1) % len(self.files))
        x = torch.from_numpy(np.transpose(arr, (3, 0, 1, 2)))  # [C,T,H,W]
        return x, label_of(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục gốc chứa video (đệ quy)")
    ap.add_argument("--backbone", default="r2plus1d_18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="weights/vio_r2plus1d.pt")
    args = ap.parse_args()

    files = []
    for ext in _EXTS:
        files += glob.glob(os.path.join(args.data, "**", ext), recursive=True)
    if not files:
        raise SystemExit(f"không thấy video trong {args.data}")
    random.seed(42)
    random.shuffle(files)
    pos = sum(label_of(f) for f in files)
    print(f"{len(files)} video — dương(violence)={pos} âm={len(files)-pos}")

    n_val = int(len(files) * args.val_split)
    val_files, train_files = files[:n_val], files[n_val:]
    tr = DataLoader(ClipDataset(train_files, args.clip_len, args.size, True),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    va = DataLoader(ClipDataset(val_files, args.clip_len, args.size, False),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_backbone(args.backbone, num_classes=2, pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tot, run = 0, 0.0
        for x, y in tr:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            tot += x.size(0)
        sched.step()

        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                n += y.size(0)
        acc = correct / max(n, 1)
        print(f"epoch {ep+1}/{args.epochs} loss={run/max(tot,1):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "clip_len": args.clip_len, "size": args.size, "val_acc": acc}, args.out)
            print(f"  ✓ saved {args.out} (val_acc={acc:.4f})")

    print(f"done. best val_acc={best:.4f} → {args.out}")


if __name__ == "__main__":
    main()
