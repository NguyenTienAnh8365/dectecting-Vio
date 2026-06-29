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
import glob
import json
import os

import cv2

from transport import config
from transport.model import VioModel
from train.train_clip import label_of

_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mpg", "*.mpeg")


def _list_videos(data: str) -> list[str]:
    out: list[str] = []
    for ext in _EXTS:
        out += glob.glob(os.path.join(data, "**", ext), recursive=True)
    return sorted(out)


def probe_video(path: str, model: VioModel, gate=None) -> list[dict]:
    """Trượt cửa sổ WIN_LEN mỗi WIN_STRIDE frame → [{prob, persons}] cho từng cửa sổ."""
    cap = cv2.VideoCapture(path)
    win: list = []
    since, out = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        win.append(frame)
        if len(win) > config.WIN_LEN:
            win.pop(0)
        since += 1
        if len(win) == config.WIN_LEN and since >= config.WIN_STRIDE:
            since = 0
            prob = model.predict(list(win))
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


def build_cache(data: str, limit: int, use_gate: bool) -> dict:
    model = VioModel()
    if model.dummy:
        raise SystemExit("[eval] model DUMMY (chưa có engine/ckpt) → prob=0, vô nghĩa. "
                         "Trỏ VIO_ENGINE/VIO_CKPT trước khi đo.")
    gate = None
    if use_gate:
        from ultralytics import YOLO
        gate = YOLO(config.GATE_MODEL)

    vids = _list_videos(data)
    if limit:  # cân bằng: lấy tối đa `limit` mỗi lớp để khỏi lệch
        pos = [v for v in vids if label_of(v) == 1][:limit]
        neg = [v for v in vids if label_of(v) == 0][:limit]
        vids = pos + neg
    print(f"[eval] {len(vids)} video, gate={'on' if use_gate else 'off'} → trích prob cửa sổ…")

    cache = {"meta": {"win_len": config.WIN_LEN, "win_stride": config.WIN_STRIDE,
                      "min_persons": config.MIN_PERSONS, "gate": use_gate}, "videos": []}
    for i, v in enumerate(vids, 1):
        windows = probe_video(v, model, gate)
        cache["videos"].append({"path": v, "label": label_of(v), "windows": windows})
        if i % 10 == 0 or i == len(vids):
            print(f"  {i}/{len(vids)}  {os.path.basename(v)} ({len(windows)} cửa sổ)")
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
    ap.add_argument("--data", help="thư mục video có nhãn (RLVS/RWF). Bỏ qua nếu --sweep-only")
    ap.add_argument("--cache", default="logs/probe.json", help="file cache prob cửa sổ")
    ap.add_argument("--limit", type=int, default=0, help="tối đa N video mỗi lớp (0=tất cả)")
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
        if not args.data:
            raise SystemExit("cần --data (hoặc --sweep-only để đọc cache có sẵn)")
        cache = build_cache(args.data, args.limit, args.gate)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print(f"[eval] đã lưu cache → {args.cache}")

    sweep(cache, _grid(args.lows), _grid(args.strongs),
          [int(x) for x in args.ks.split(",")])


if __name__ == "__main__":
    main()
