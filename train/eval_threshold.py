"""Đo ngưỡng (threshold) trên video CÓ NHÃN → chọn T_low/T_strong/K có cơ sở dữ liệu.

Vì sao cần: ngưỡng cứng 0.6/0.7 chọn bằng cảm tính dễ bỏ sót bạo lực ngắn hoặc báo nhầm.
Công cụ này chạy classifier theo ĐÚNG cửa sổ trượt như lúc chạy thật (WIN_LEN/WIN_STRIDE),
lưu prob từng cửa sổ ra cache, rồi quét nhiều bộ ngưỡng IN-MEMORY (không infer lại) để in
precision/recall/F1 — khớp chính xác logic verdict ở transport/analyze.py + dashboard:

    violence = (số cửa sổ prob≥T_low ≥ K)  OR  (có ≥1 cửa sổ prob≥T_strong)

Nhãn suy từ đường dẫn (dùng lại train.train_clip.label_of): RLVS Violence/NonViolence,
RWF-2000 Fight/NonFight.

Dùng:
  # B1: trích prob từng cửa sổ (1 lần, chậm — có GPU/engine càng nhanh), cache ra JSON
  python -m train.eval_threshold --data data/RLVS --limit 150 --cache logs/probe.json
  # B2: đổi lưới ngưỡng tùy ý — đọc cache, quét tức thì (không cần GPU)
  python -m train.eval_threshold --cache logs/probe.json --sweep-only

Mặc định KHÔNG qua person-gate (đo riêng tín hiệu classifier cho gọn). Thêm --gate để mô
phỏng đầy đủ điều kiện MIN_PERSONS như chạy thật.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import cv2

from transport import config
from transport.model import VioModel, even_indices

_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mpg", "*.mpeg")


def label_of(path: str) -> int:
    """Nhãn từ đường dẫn (chỉ cho chế độ --data; --manifest đã có nhãn sẵn)."""
    p = path.lower()
    neg = ("non" in p) or ("normal" in p)
    pos = ("fight" in p) or ("viol" in p)
    return 1 if (pos and not neg) else 0


def _list_videos(data: str) -> list[str]:
    out: list[str] = []
    for ext in _EXTS:
        out += glob.glob(os.path.join(data, "**", ext), recursive=True)
    return sorted(out)


def probe_video(path: str, model: VioModel, gate=None, crop_box=None) -> list[dict]:
    """Trượt cửa sổ WIN_LEN mỗi WIN_STRIDE frame → [{prob, persons}] cho từng cửa sổ.

    crop_box=(x1,y1,x2,y2): crop vùng người (khớp infer/train). None = cả khung.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    span = max(config.WIN_LEN, round(config.WIN_SECONDS * fps))  # buffer span thời gian → lấy đều 16
    win: list = []
    since, out = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if crop_box:
            x1, y1, x2, y2 = crop_box
            frame = frame[y1:y2, x1:x2]
        win.append(frame)
        if len(win) > span:
            win.pop(0)
        since += 1
        if len(win) >= config.WIN_LEN and since >= config.WIN_STRIDE:
            since = 0
            # MULTI-SPAN: max prob qua các span (frames đã pre-crop theo crop_box cả-clip).
            n = len(win)
            prob = 0.0
            for sf in sorted({min(max(round(s * fps), config.WIN_LEN), n) for s in config.WIN_SECONDS_LIST}):
                seg = win[n - sf:]
                sel = even_indices(len(seg), config.WIN_LEN)
                prob = max(prob, model.predict([seg[i] for i in sel]))
            persons = config.MIN_PERSONS  # mặc định coi như đủ người (gate tắt)
            if gate is not None:
                res = gate.predict(frame, imgsz=config.GATE_IMGSZ, conf=config.GATE_CONF,
                                   classes=[config.PERSON_CLASS], quantize=config.GATE_QUANTIZE,
                                   device=config.GATE_DEVICE, verbose=False)[0]
                b = res.boxes
                persons = 0 if b is None else int((b.cls.cpu().numpy().astype(int)
                                                   == config.PERSON_CLASS).sum())
            out.append({"prob": round(float(prob), 4), "persons": int(persons)})
    cap.release()
    return out


def _rows_from_manifest(manifest: str, split: str) -> list[tuple[str, int]]:
    root = os.path.join(os.path.dirname(__file__), os.pardir)
    out = []
    with open(manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                out.append((os.path.join(root, r["path"]), int(r["label"])))
    return out


def build_cache(rows: list[tuple[str, int]], use_gate: bool, crop_boxes: dict) -> dict:
    model = VioModel()
    if model.dummy:
        raise SystemExit("[eval] model DUMMY (chưa có engine/ckpt) → prob=0, vô nghĩa. "
                         "Trỏ VIO_ENGINE/VIO_CKPT trước khi đo.")
    gate = None
    if use_gate:
        from ultralytics import YOLO
        gate = YOLO(config.GATE_MODEL)

    root = os.path.join(os.path.dirname(__file__), os.pardir)
    print(f"[eval] {len(rows)} video, gate={'on' if use_gate else 'off'}, "
          f"crop={'on' if crop_boxes else 'off'} → trích prob cửa sổ…")
    cache = {"meta": {"win_len": config.WIN_LEN, "win_stride": config.WIN_STRIDE,
                      "min_persons": config.MIN_PERSONS, "gate": use_gate}, "videos": []}
    for i, (path, label) in enumerate(rows, 1):
        rel = os.path.relpath(path, root)
        info = crop_boxes.get(rel) or {}
        box = tuple(info["box"]) if info.get("box") else None
        windows = probe_video(path, model, gate, crop_box=box)
        cache["videos"].append({"path": path, "label": label, "windows": windows})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  {os.path.basename(path)} ({len(windows)} cửa sổ)")
    return cache


def _verdict(windows: list[dict], t_low: float, t_strong: float, k: int, min_persons: int) -> bool:
    pos = sum(1 for w in windows if w["prob"] >= t_low and w["persons"] >= min_persons)
    strong = any(w["prob"] >= t_strong and w["persons"] >= min_persons for w in windows)
    return pos >= k or strong


def sweep(cache: dict, lows, strongs, ks) -> None:
    vids = cache["videos"]
    min_persons = cache["meta"].get("min_persons", config.MIN_PERSONS)
    n_pos = sum(1 for v in vids if v["label"] == 1)
    n_neg = len(vids) - n_pos
    print(f"\n[sweep] {len(vids)} video ({n_pos} bạo lực / {n_neg} thường)")
    print(f"{'T_low':>6} {'T_str':>6} {'K':>3} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} | "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 70)

    rows = []
    for t_low in lows:
        for t_strong in strongs:
            if t_strong < t_low:
                continue
            for k in ks:
                tp = fp = fn = tn = 0
                for v in vids:
                    pred = _verdict(v["windows"], t_low, t_strong, k, min_persons)
                    if v["label"] == 1:
                        tp += pred; fn += not pred
                    else:
                        fp += pred; tn += not pred
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                rows.append((t_low, t_strong, k, tp, fp, fn, tn, prec, rec, f1))
                print(f"{t_low:6.2f} {t_strong:6.2f} {k:3d} | {tp:4d} {fp:4d} {fn:4d} {tn:4d} | "
                      f"{prec:6.3f} {rec:6.3f} {f1:6.3f}")

    best_f1 = max(rows, key=lambda r: r[9])
    best_rec = max(rows, key=lambda r: (r[8], r[7]))  # recall cao nhất, ưu tiên ít FP
    print("-" * 70)
    print(f"⭐ F1 cao nhất : T_low={best_f1[0]} T_strong={best_f1[1]} K={best_f1[2]} "
          f"→ P={best_f1[7]:.3f} R={best_f1[8]:.3f} F1={best_f1[9]:.3f}")
    print(f"⭐ Recall cao  : T_low={best_rec[0]} T_strong={best_rec[1]} K={best_rec[2]} "
          f"→ P={best_rec[7]:.3f} R={best_rec[8]:.3f} F1={best_rec[9]:.3f}  "
          f"(ưu tiên không bỏ sót)")


def _grid(s: str) -> list[float]:
    return [round(float(x), 3) for x in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="thư mục video có nhãn (RLVS/RWF). Bỏ qua nếu --manifest/--sweep-only")
    ap.add_argument("--manifest", help="dùng manifest.csv (ưu tiên hơn --data)")
    ap.add_argument("--split", default="test", help="split của manifest (mặc định test=VioPeru)")
    ap.add_argument("--crop-boxes", default="data/cache/crop_boxes.json",
                    help="cache crop-to-person (khớp infer); rỗng = không crop")
    ap.add_argument("--cache", default="logs/probe.json", help="file cache prob cửa sổ")
    ap.add_argument("--limit", type=int, default=0, help="tối đa N video mỗi lớp (0=tất cả; chỉ --data)")
    ap.add_argument("--gate", action="store_true", help="chạy person-gate (mô phỏng MIN_PERSONS)")
    ap.add_argument("--sweep-only", action="store_true", help="chỉ đọc cache + quét, không infer")
    ap.add_argument("--lows", default="0.45,0.55,0.60,0.65,0.70", help="lưới T_low")
    ap.add_argument("--strongs", default="0.80,0.85,0.90,0.95", help="lưới T_strong")
    ap.add_argument("--ks", default="1,2,3", help="lưới K (số cửa sổ dương)")
    args = ap.parse_args()

    if args.sweep_only:
        with open(args.cache, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        if args.manifest:
            rows = _rows_from_manifest(args.manifest, args.split)
        elif args.data:
            vids = _list_videos(args.data)
            if args.limit:
                pos = [v for v in vids if label_of(v) == 1][:args.limit]
                neg = [v for v in vids if label_of(v) == 0][:args.limit]
                vids = pos + neg
            rows = [(v, label_of(v)) for v in vids]
        else:
            raise SystemExit("cần --manifest hoặc --data (hoặc --sweep-only để đọc cache có sẵn)")
        crop_boxes = {}
        if args.crop_boxes and os.path.exists(args.crop_boxes):
            crop_boxes = json.load(open(args.crop_boxes, encoding="utf-8"))
        cache = build_cache(rows, args.gate, crop_boxes)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"[eval] đã lưu cache → {args.cache}")

    sweep(cache, _grid(args.lows), _grid(args.strongs),
          [int(x) for x in args.ks.split(",")])


if __name__ == "__main__":
    main()
