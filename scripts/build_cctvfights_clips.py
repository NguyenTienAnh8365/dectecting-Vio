"""Trim CCTV-Fights (ROSE Lab NTU, cần agreement) → data/CCTVFights/{Fight,NonFight}.

Nguồn: data/_dl/cctvfights/ (5 zip original_format đã giải nén + groundtruth.json kiểu
ActivityNet: mỗi video có annotations[].segment=[t0,t1] giây, label='Fight', người gán tay).
1000 video untrimmed (280 CCTV / 707 mobile), 2414 segment fight, median 4.4s.

Luật (rút kinh nghiệm UBI/UCF — xem README "KẾT QUẢ đợt mở rộng"):
- Positive: trim TRỌN từng segment Fight (giữ nguyên đoạn gốc; chỉ chia khi >MAX_S để
  16-frame-trải-đều không loãng); bỏ segment <MIN_S.
- Negative "cùng camera": vùng ngoài segment, đệm MARGIN_S khỏi biên (chống biên nhãn hở),
  chunk CLIP_S, tối đa bằng số positive của video. PHẢI audit bằng ckpt cũ trước khi train.

Chạy: `uv run python scripts/build_cctvfights_clips.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from imageio_ffmpeg import get_ffmpeg_exe

DL = os.path.join(os.path.dirname(__file__), os.pardir, "data", "_dl", "cctvfights")
OUT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "CCTVFights")
MIN_S = 1.5    # segment/chunk ngắn hơn thì bỏ
MAX_S = 10.0   # positive dài hơn thì chia (16 frame trải đều quá loãng)
CLIP_S = 5.0   # độ dài chunk negative
MARGIN_S = 2.5
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
        print(f"  LỖI {os.path.basename(dst)}: {r.stderr.decode()[:150]}")
        try:
            os.remove(dst)
        except OSError:
            pass
        return False
    return True


def pieces(t0: float, t1: float, max_s: float, min_s: float) -> list[tuple[float, float]]:
    """Chia [t0,t1) thành đoạn ≤max_s (đoạn cuối gộp nếu <min_s)."""
    out = []
    t = t0
    while t1 - t > max_s + min_s:
        out.append((t, max_s))
        t += max_s
    if t1 - t >= min_s:
        out.append((t, t1 - t))
    return out


def main() -> None:
    os.makedirs(os.path.join(OUT, "Fight"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "NonFight"), exist_ok=True)
    db = json.load(open(os.path.join(DL, "groundtruth.json")))["database"]

    src_of = {}
    for f in os.listdir(DL):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".zip", ".json"):
            src_of[stem] = os.path.join(DL, f)

    jobs, n_miss = [], 0
    for stem, meta in sorted(db.items()):
        src = src_of.get(stem)
        if not src:
            n_miss += 1
            continue
        dur = float(meta["duration"])
        spans = sorted((float(a), min(float(b), dur)) for a, b in
                       (s["segment"] for s in meta["annotations"]))
        n_pos = 0
        for t0, t1 in spans:
            for t, d in pieces(t0, t1, MAX_S, MIN_S):
                jobs.append((src, os.path.join(OUT, "Fight", f"{stem}_p{n_pos:02d}.mp4"), t, d))
                n_pos += 1
        gaps, prev = [], 0.0
        for t0, t1 in spans:
            gaps.append((prev, t0))
            prev = t1
        gaps.append((prev, dur))
        n_neg = 0
        for g0, g1 in gaps:
            for t, d in pieces(g0 + MARGIN_S, g1 - MARGIN_S, CLIP_S, MIN_S + 0.5):
                if n_neg >= n_pos:
                    break
                jobs.append((src, os.path.join(OUT, "NonFight", f"{stem}_n{n_neg:02d}.mp4"), t, d))
                n_neg += 1

    n_pos = sum(1 for j in jobs if os.sep + "Fight" + os.sep in j[1])
    print(f"{n_miss} video thiếu file; cắt {len(jobs)} clip "
          f"({n_pos} Fight / {len(jobs)-n_pos} NonFight)…")
    with ThreadPoolExecutor(16) as ex:
        oks = list(ex.map(lambda j: cut(*j), jobs))
    print(f"done: {sum(oks)}/{len(jobs)} OK → {OUT}")


if __name__ == "__main__":
    main()
