"""Cắt UBI-Fights (mirror Kaggle `intissarziani/ubi-fightsall`) → clip 3-7s ở data/UBIFights/{Fight,NonFight}.

Bộ gốc: 1000 video CCTV (216 fight / 784 normal), nhãn 0/1 TỪNG FRAME (annotation/<stem>.csv).
Trang gốc socia-lab.di.ubi.pt chết (2026-07) → dùng mirror Kaggle (tải bằng kagglehub, không cần key).

Luật cắt:
- Video fight: mỗi run frame=1 cắt thành chunk CLIP_S giây (giữ chunk ≥ MIN_S).
- Negative "cùng bối cảnh": run frame=0 của CHÍNH video fight (đệm MARGIN_S khỏi biên fight),
  tối đa bằng số chunk positive của video đó — loại negative giá trị nhất (cùng khung camera).
- Video normal (N_*): lấy NEG_PER_NORMAL clip đều nhau — tránh ngập negative.

Chạy: `uv run python scripts/build_ubifights_clips.py`  (song song 16 ffmpeg).
"""
from __future__ import annotations

import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

SRC = os.environ.get(
    "UBI_SRC",
    os.path.expanduser("~/.cache/kagglehub/datasets/intissarziani/ubi-fightsall/versions/1/UBI_FIGHTS"),
)
OUT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "UBIFights")
CLIP_S = 5.0    # độ dài chunk chuẩn
MIN_S = 2.0     # bỏ chunk ngắn hơn
MARGIN_S = 2.0  # negative cùng video phải cách biên fight chừng này
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


def runs_of(labels: list[int], want: int) -> list[tuple[int, int]]:
    """[(start,end)) các đoạn frame liên tiếp có nhãn `want`."""
    out, s = [], None
    for i, v in enumerate(labels):
        if v == want and s is None:
            s = i
        elif v != want and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(labels)))
    return out


def chunks(t0: float, t1: float, cap: int = 12) -> list[tuple[float, float]]:
    """Chia [t0,t1) giây thành chunk CLIP_S, giữ ≥MIN_S, tối đa `cap`."""
    out = []
    t = t0
    while t + MIN_S <= t1 and len(out) < cap:
        d = min(CLIP_S, t1 - t)
        out.append((t, d))
        t += CLIP_S
    return out


def fps_of(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps if fps > 1 else 30.0


def jobs_for_fight(vid: str) -> list[tuple[str, str, float, float]]:
    stem = os.path.splitext(os.path.basename(vid))[0]
    ann = os.path.join(SRC, "annotation", stem + ".csv")
    if not os.path.exists(ann):
        print(f"  thiếu annotation {stem} — bỏ")
        return []
    labels = [int(float(x)) for x in open(ann).read().split()]
    fps = fps_of(vid)
    jobs = []
    n_pos = 0
    for s, e in runs_of(labels, 1):
        for t, d in chunks(s / fps, e / fps):
            jobs.append((vid, os.path.join(OUT, "Fight", f"{stem}_p{n_pos:02d}.mp4"), t, d))
            n_pos += 1
    # negative cùng bối cảnh (đệm MARGIN_S), tối đa n_pos chunk
    n_neg = 0
    for s, e in runs_of(labels, 0):
        t0, t1 = s / fps + MARGIN_S, e / fps - MARGIN_S
        for t, d in chunks(t0, t1):
            if n_neg >= n_pos:
                break
            jobs.append((vid, os.path.join(OUT, "NonFight", f"{stem}_n{n_neg:02d}.mp4"), t, d))
            n_neg += 1
    return jobs


def jobs_for_normal(vid: str, rng: random.Random) -> list[tuple[str, str, float, float]]:
    stem = os.path.splitext(os.path.basename(vid))[0]
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    dur = n / fps if fps > 1 else 0
    if dur < MIN_S + 1:
        return []
    jobs = []
    for i in range(NEG_PER_NORMAL):
        lo = dur * i / NEG_PER_NORMAL
        hi = max(lo, dur * (i + 1) / NEG_PER_NORMAL - CLIP_S)
        t = rng.uniform(lo, hi)
        d = min(CLIP_S, dur - t)
        if d >= MIN_S:
            jobs.append((vid, os.path.join(OUT, "NonFight", f"{stem}_n{i:02d}.mp4"), t, d))
    return jobs


def main() -> None:
    os.makedirs(os.path.join(OUT, "Fight"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "NonFight"), exist_ok=True)
    rng = random.Random(42)
    fights = sorted(
        os.path.join(SRC, "videos", "fight", f)
        for f in os.listdir(os.path.join(SRC, "videos", "fight")) if f.endswith(".mp4"))
    normals = sorted(
        os.path.join(SRC, "videos", "normal", f)
        for f in os.listdir(os.path.join(SRC, "videos", "normal")) if f.endswith(".mp4"))
    print(f"nguồn: {len(fights)} fight / {len(normals)} normal video")

    jobs: list[tuple[str, str, float, float]] = []
    for v in fights:
        jobs += jobs_for_fight(v)
    for v in normals:
        jobs += jobs_for_normal(v, rng)
    n_pos = sum(1 for j in jobs if os.sep + "Fight" + os.sep in j[1])
    print(f"cắt {len(jobs)} clip ({n_pos} Fight / {len(jobs)-n_pos} NonFight)…")

    with ThreadPoolExecutor(16) as ex:
        oks = list(ex.map(lambda j: cut(*j), jobs))
    print(f"done: {sum(oks)}/{len(jobs)} OK → {OUT}")


if __name__ == "__main__":
    main()
