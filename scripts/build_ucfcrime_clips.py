"""Cắt UCF-Crime (chỉ Fighting/Assault/Abuse + Normal) → clip 3-7s ở data/UCFCrime/{Fight,Normal}.

Nguồn: HF `jinmang2/ucf_crime` (zip gốc CRCV) đã giải nén lớp liên quan vào data/_dl/ucfcrime/raw/.
Annotation temporal (giây) lấy từ HF `backseollgi/UCF_Crime_ICML_annotation` — phủ CẢ train
lẫn test split (annotation chính thức CRCV chỉ có test); trường `events` = [[t0,t1],…] giây.

Luật cắt:
- Video anomaly: mỗi event cắt chunk CLIP_S giây (giữ ≥MIN_S, cap CAP_PER_EVENT tránh video
  dài nuốt phân phối) → Fight.
- Negative "cùng bối cảnh": vùng ngoài event (đệm MARGIN_S), tối đa bằng số positive của video.
- Video Normal_*: NEG_PER_NORMAL clip đều nhau/video.

Chạy: `uv run python scripts/build_ucfcrime_clips.py`.
"""
from __future__ import annotations

import glob
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

DL = os.path.join(os.path.dirname(__file__), os.pardir, "data", "_dl", "ucfcrime")
RAW = os.path.join(DL, "raw")
OUT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "UCFCrime")
CLASSES = ("Fighting", "Assault", "Abuse")
CLIP_S = 5.0
MIN_S = 2.0
MARGIN_S = 3.0
CAP_PER_EVENT = 8
NEG_PER_NORMAL = 2
FFMPEG = get_ffmpeg_exe()


def cut(src: str, dst: str, start: float, dur: float) -> bool:
    if os.path.exists(dst):
        return True
    r = subprocess.run(
        [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an",
         "-loglevel", "error", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  LỖI cắt {os.path.basename(dst)}: {r.stderr.decode()[:200]}")
        try:
            os.remove(dst)
        except OSError:
            pass
        return False
    return True


def chunks(t0: float, t1: float, cap: int) -> list[tuple[float, float]]:
    out = []
    t = t0
    while t + MIN_S <= t1 and len(out) < cap:
        out.append((t, min(CLIP_S, t1 - t)))
        t += CLIP_S
    return out


def video_dur(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps if fps > 1 else 0.0


def load_events() -> dict[str, list[tuple[float, float]]]:
    """stem (vd Fighting008_x264) → [(t0,t1)] giây, gộp json train+test."""
    ev: dict[str, list[tuple[float, float]]] = {}
    for f in glob.glob(os.path.join(DL, "ucf_database_*_category.json")):
        for stem, d in json.load(open(f)).items():
            spans = [(float(a), float(b)) for a, b in d.get("events", []) if b > a]
            if spans:
                ev.setdefault(stem, []).extend(spans)
    return ev


def main() -> None:
    os.makedirs(os.path.join(OUT, "Fight"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "Normal"), exist_ok=True)
    events = load_events()
    jobs: list[tuple[str, str, float, float]] = []
    n_no_ann = 0

    for cls in CLASSES:
        vids = sorted(glob.glob(os.path.join(RAW, "Anomaly-Videos-*", cls, "*.mp4")))
        for vid in vids:
            stem = os.path.splitext(os.path.basename(vid))[0]
            spans = sorted(events.get(stem, []))
            if not spans:
                n_no_ann += 1
                continue
            dur = video_dur(vid)
            n_pos = 0
            for t0, t1 in spans:
                for t, d in chunks(t0, min(t1, dur), CAP_PER_EVENT):
                    jobs.append((vid, os.path.join(OUT, "Fight", f"{stem}_p{n_pos:02d}.mp4"), t, d))
                    n_pos += 1
            # negative cùng bối cảnh: vùng ngoài event, đệm MARGIN_S
            gaps, prev = [], 0.0
            for t0, t1 in spans:
                gaps.append((prev, t0))
                prev = t1
            gaps.append((prev, dur))
            n_neg = 0
            for g0, g1 in gaps:
                for t, d in chunks(g0 + MARGIN_S, g1 - MARGIN_S, CAP_PER_EVENT):
                    if n_neg >= n_pos:
                        break
                    jobs.append((vid, os.path.join(OUT, "Normal", f"{stem}_n{n_neg:02d}.mp4"), t, d))
                    n_neg += 1

    rng = random.Random(42)
    normals = sorted(
        glob.glob(os.path.join(RAW, "Testing_Normal_Videos_Anomaly", "*.mp4"))
        + glob.glob(os.path.join(RAW, "Normal_Videos_for_Event_Recognition", "*.mp4")))
    for vid in normals:
        stem = os.path.splitext(os.path.basename(vid))[0]
        dur = video_dur(vid)
        if dur < MIN_S + 1:
            continue
        for i in range(NEG_PER_NORMAL):
            lo = dur * i / NEG_PER_NORMAL
            hi = max(lo, dur * (i + 1) / NEG_PER_NORMAL - CLIP_S)
            t = rng.uniform(lo, hi)
            d = min(CLIP_S, dur - t)
            if d >= MIN_S:
                jobs.append((vid, os.path.join(OUT, "Normal", f"{stem}_n{i:02d}.mp4"), t, d))

    n_pos = sum(1 for j in jobs if os.sep + "Fight" + os.sep in j[1])
    print(f"{n_no_ann} video anomaly thiếu annotation (bỏ); "
          f"cắt {len(jobs)} clip ({n_pos} Fight / {len(jobs)-n_pos} Normal)…")
    with ThreadPoolExecutor(16) as ex:
        oks = list(ex.map(lambda j: cut(*j), jobs))
    print(f"done: {sum(oks)}/{len(jobs)} OK → {OUT}")


if __name__ == "__main__":
    main()
