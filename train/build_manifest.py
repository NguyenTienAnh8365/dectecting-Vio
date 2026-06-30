"""Quét 5 dataset → data/manifest.csv (cột path,label,dataset,split). Nguồn chân lý cho train.

Thay heuristic chuỗi `label_of` (sai với "Weaponized") bằng LUẬT TỪNG DATASET rõ ràng. Split:
- VioPeru → toàn bộ `test` (cross-domain, KHÔNG bao giờ train) — đo tổng quát hóa sang CCTV thật.
- 4 bộ còn lại → stratified theo lớp, theo từng dataset (seed cố định) thành train/val.
SCVD chỉ có Violence/Normal (Weaponized đã bị loại từ khâu tái dựng) → khớp quyết định.

Chạy: `uv run python -m train.build_manifest`  (tùy chọn --val-frac, --seed, --out).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import random
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
DATA = os.path.join(ROOT, "data")
EXTS = (".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv")


def _vids(*parts) -> list[str]:
    base = os.path.join(DATA, *parts)
    out = []
    for e in EXTS:
        out += glob.glob(os.path.join(base, "**", "*" + e), recursive=True)
        out += glob.glob(os.path.join(base, "**", "*" + e.upper()), recursive=True)
    return sorted(set(out))


def _has(path: str, *names: str) -> bool:
    """True nếu path chứa 1 trong các thư mục `names` (so khớp theo phân đoạn, hoa/thường)."""
    segs = [s.lower() for s in path.replace("\\", "/").split("/")]
    return any(n.lower() in segs for n in names)


def collect() -> list[tuple[str, int, str]]:
    """→ list (path, label, dataset). Bỏ qua file không khớp luật để tránh nhãn sai."""
    rows: list[tuple[str, int, str]] = []

    # RLVS: Violence / NonViolence
    for p in _vids("RLVS"):
        if _has(p, "NonViolence"):
            rows.append((p, 0, "RLVS"))
        elif _has(p, "Violence"):
            rows.append((p, 1, "RLVS"))

    # RWF-2000: .../Fight/ vs .../NonFight/ (gộp train+val của bộ vào pool, ta tự split lại)
    for p in _vids("RWF2000"):
        if _has(p, "NonFight"):
            rows.append((p, 0, "RWF2000"))
        elif _has(p, "Fight"):
            rows.append((p, 1, "RWF2000"))

    # Surveillance Fight: fight / noFight
    for p in _vids("SurveillanceFight"):
        if _has(p, "noFight"):
            rows.append((p, 0, "SurveillanceFight"))
        elif _has(p, "fight"):
            rows.append((p, 1, "SurveillanceFight"))

    # SCVD: Violence / Normal (Weaponized đã loại lúc tái dựng)
    for p in _vids("SCVD"):
        if _has(p, "Normal"):
            rows.append((p, 0, "SCVD"))
        elif _has(p, "Violence"):
            rows.append((p, 1, "SCVD"))

    # VioPeru: Fight / NonFight + false_positives_validation(→0). TẤT CẢ là test.
    for p in _vids("VioPeru"):
        if _has(p, "false_positives_validation") or _has(p, "NonFight"):
            rows.append((p, 0, "VioPeru"))
        elif _has(p, "Fight"):
            rows.append((p, 1, "VioPeru"))

    return rows


def split(rows, val_frac: float, seed: int) -> list[tuple[str, int, str, str]]:
    """VioPeru→test; còn lại stratified theo (dataset,label) → train/val."""
    rng = random.Random(seed)
    out: list[tuple[str, int, str, str]] = []
    buckets: dict[tuple[str, int], list[str]] = {}
    for p, y, ds in rows:
        if ds == "VioPeru":
            out.append((p, y, ds, "test"))
        else:
            buckets.setdefault((ds, y), []).append(p)
    for (ds, y), paths in buckets.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_frac))
        for i, p in enumerate(paths):
            out.append((p, y, ds, "val" if i < n_val else "train"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(DATA, "manifest.csv"))
    args = ap.parse_args()

    rows = collect()
    full = split(rows, args.val_frac, args.seed)
    full.sort(key=lambda r: (r[3], r[2], r[1], r[0]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "dataset", "split"])
        for p, y, ds, sp in full:
            w.writerow([os.path.relpath(p, ROOT), y, ds, sp])

    # Thống kê + hash (đưa vào checkpoint để truy vết)
    print(f"→ {args.out}  ({len(full)} clip)")
    by = Counter((sp, ds, y) for _, y, ds, sp in full)
    for sp in ("train", "val", "test"):
        print(f"\n[{sp}]")
        dss = sorted({ds for (s, ds, y) in by if s == sp})
        for ds in dss:
            pos = by[(sp, ds, 1)]
            neg = by[(sp, ds, 0)]
            print(f"  {ds:18s} pos={pos:5d} neg={neg:5d} tổng={pos+neg}")
        tot_pos = sum(v for (s, ds, y), v in by.items() if s == sp and y == 1)
        tot_neg = sum(v for (s, ds, y), v in by.items() if s == sp and y == 0)
        print(f"  {'TỔNG':18s} pos={tot_pos:5d} neg={tot_neg:5d} tổng={tot_pos+tot_neg}")

    h = hashlib.sha1(open(args.out, "rb").read()).hexdigest()[:12]
    print(f"\nmanifest_sha1={h}")


if __name__ == "__main__":
    main()
