"""Quét 7 dataset → data/manifest.csv (cột path,label,dataset,split). Nguồn chân lý cho train.

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


def _excluded() -> set[str]:
    """data/exclude_clips.txt: clip bị loại sau audit nhãn (Abuse tĩnh, negative 'cùng bối cảnh'
    mà model tin cậy chấm prob cao = nghi lẫn bạo lực). Mỗi dòng 1 path tương đối gốc repo."""
    p = os.path.join(DATA, "exclude_clips.txt")
    if not os.path.exists(p):
        return set()
    return {ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()}


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


def _group_key(path: str, ds: str) -> str:
    """Đơn vị split = VIDEO NGUỒN. Với dataset dạng chunk cắt từ video dài (đặt tên
    <stem>_pNN/_nNN — vd footage CCTV địa phương sau này) gom theo stem gốc, không thì
    chunk cùng video lọt cả train lẫn val (leak, val ảo cao). Bài học đợt UBI/UCF 2026-07."""
    stem = os.path.splitext(os.path.basename(path))[0]
    import re
    return re.sub(r"_[pn]\d+$", "", stem)


def _vioperu_group(path: str) -> str:
    """Cụm camera VioPeru từ tên file (nf100and.avi / val_nfsanj01.avi → and/sanj/tala)."""
    import re
    m = re.search(r"(and|sanj|tala)", os.path.basename(path))
    return m.group(1) if m else ""


def split(rows, val_frac: float, seed: int,
          vioperu_train_groups: frozenset[str] = frozenset()) -> list[tuple[str, int, str, str]]:
    """VioPeru→test; còn lại stratified theo (dataset,label) → train/val, đơn vị = video nguồn.

    vioperu_train_groups: cụm camera VioPeru (and/sanj/tala) đưa vào TRAIN (finetune in-domain);
    cụm còn lại vẫn test — thước đo "camera chưa thấy". Rỗng = VioPeru toàn bộ test (mặc định cũ)."""
    rng = random.Random(seed)
    out: list[tuple[str, int, str, str]] = []
    buckets: dict[tuple[str, int], dict[str, list[str]]] = {}
    group_split: dict[tuple[str, str], str] = {}  # (ds, group) → 'train'/'val' (nhất quán cả 2 nhãn)
    for p, y, ds in rows:
        if ds == "VioPeru":
            if _vioperu_group(p) in vioperu_train_groups:
                # đi qua bucket thường → có mặt cả train lẫn val (early-stop thấy tín hiệu in-domain)
                buckets.setdefault((ds, y), {}).setdefault(_group_key(p, ds), []).append(p)
            else:
                out.append((p, y, ds, "test"))
        else:
            buckets.setdefault((ds, y), {}).setdefault(_group_key(p, ds), []).append(p)
    for (ds, y), groups in sorted(buckets.items()):
        names = sorted(groups)
        rng.shuffle(names)
        n_clips = sum(len(groups[g]) for g in names)
        want_val = max(1, int(n_clips * val_frac))
        got_val = 0
        for g in names:
            sp = group_split.get((ds, g))  # video nguồn đã gán ở nhãn kia → giữ nguyên phía
            if sp is None:
                sp = "val" if got_val < want_val else "train"
                group_split[(ds, g)] = sp
            if sp == "val":
                got_val += len(groups[g])
            for p in sorted(groups[g]):
                out.append((p, y, ds, sp))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(DATA, "manifest.csv"))
    ap.add_argument("--vioperu-train-groups", default="",
                    help="cụm camera VioPeru đưa vào train, vd 'sanj,tala' (cụm còn lại vẫn test)")
    ap.add_argument("--skip-datasets", default="",
                    help="bỏ hẳn dataset khỏi manifest, vd 'UBIFights,UCFCrime'")
    args = ap.parse_args()

    rows = collect()
    if args.skip_datasets:
        skip = {s.strip() for s in args.skip_datasets.split(",") if s.strip()}
        rows = [r for r in rows if r[2] not in skip]
    excl = _excluded()
    if excl:
        n0 = len(rows)
        rows = [r for r in rows if os.path.relpath(r[0], ROOT) not in excl]
        print(f"[exclude] loại {n0 - len(rows)}/{len(excl)} clip theo data/exclude_clips.txt")
    vp_groups = frozenset(s.strip() for s in args.vioperu_train_groups.split(",") if s.strip())
    full = split(rows, args.val_frac, args.seed, vp_groups)
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
