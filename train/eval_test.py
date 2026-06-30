"""Đánh giá clip-level một checkpoint trên 1 split của manifest (mặc định test = VioPeru).

Dùng ĐÚNG pipeline ClipDataset như training (crop-to-person từ cache, KHÔNG augment) → con số
P/R/F1/AUC so sánh trực tiếp được với val lúc train. test=VioPeru = CCTV thật chưa từng train →
đo tổng quát hóa cross-domain (sát domain Như Quỳnh nhất).

Dùng:
  uv run python -m train.eval_test --ckpt weights/vio_mvit_v2_s.pt          # test (VioPeru)
  uv run python -m train.eval_test --ckpt weights/vio_mvit_v2_s.pt --split val
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from transport.model import build_backbone, spec_of
from train.train_clip import ROOT, ClipDataset, binary_metrics, load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/vio_mvit_v2_s.pt")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--crop-boxes", default=os.path.join(ROOT, "data", "cache", "crop_boxes.json"))
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    backbone = ck.get("backbone", "mvit_v2_s")
    clip_len = ck.get("clip_len", spec_of(backbone)["clip_len"])
    crop_train = ck.get("crop_to_person", True)
    model = build_backbone(backbone, num_classes=2, pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(args.device).eval()
    print(f"[eval] {args.ckpt} backbone={backbone} clip_len={clip_len} "
          f"crop_to_person(train)={crop_train} | val lúc train={ck.get('val')}")

    crop_boxes = {}
    if not args.no_crop and os.path.exists(args.crop_boxes):
        crop_boxes = json.load(open(args.crop_boxes, encoding="utf-8"))

    rows = load_manifest(args.manifest, args.split)
    ds = ClipDataset(rows, backbone, clip_len, False, crop_boxes, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(args.device, non_blocking=True)
            with torch.autocast("cuda"):
                p = torch.softmax(model(x).float(), dim=1)[:, 1]
            probs += p.cpu().tolist()
            labels += y.tolist()

    m = binary_metrics(probs, labels)
    pos = sum(labels)
    print(f"\n[{args.split}] {len(labels)} clip (pos={pos} neg={len(labels)-pos})")
    print(f"  acc={m['acc']:.4f}  P={m['prec']:.4f}  R={m['rec']:.4f}  "
          f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}")


if __name__ == "__main__":
    main()
