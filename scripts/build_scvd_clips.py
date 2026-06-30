"""Tái dựng SCVD (HF mirror `henriquequeirozcunha/scvd-violence-5fps`) từ parquet-frame → mp4 clip.

Mirror lưu frame JPG @5fps trong parquet, kèm cột (video_id, frame_id, label). Pipeline của
ta đọc *video file*, nên gom frame theo video_id, sort theo frame_id, ghi thành 1 mp4/clip ở
data/SCVD/{Violence,Normal}/<video_id>.mp4. Bộ này đã 2 lớp (Weaponized đã bị loại từ nguồn) →
khớp quyết định "loại Weaponized". Chạy: `uv run python scripts/build_scvd_clips.py`.
"""
from __future__ import annotations

import io
import os

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "henriquequeirozcunha/scvd-violence-5fps"
FILES = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]
OUT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "SCVD")
FPS = 5.0  # sampling_fps của mirror


def _collect() -> dict:
    """video_id -> {'label': int, 'frames': {frame_id: jpg_bytes}}"""
    vids: dict = {}
    for rel in FILES:
        p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir="data/_dl/scvd")
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=512):
            for r in batch.to_pylist():
                vid = r["video_id"]
                d = vids.setdefault(vid, {"label": int(r["label"]), "frames": {}})
                img = r["image"]
                b = img["bytes"] if isinstance(img, dict) else img
                d["frames"][int(r["frame_id"])] = b
        print(f"  đọc {rel} → tổng {len(vids)} video tới giờ")
    return vids


def main() -> None:
    vids = _collect()
    n_ok, n_skip = 0, 0
    for vid, d in vids.items():
        cls = "Violence" if d["label"] == 1 else "Normal"
        outdir = os.path.join(OUT, cls)
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, f"{vid}.mp4")
        if os.path.exists(dst):
            n_ok += 1
            continue
        order = sorted(d["frames"])
        if len(order) < 4:  # quá ngắn, bỏ
            n_skip += 1
            continue
        try:
            w = imageio.get_writer(dst, fps=FPS, codec="libx264",
                                   macro_block_size=None, ffmpeg_log_level="error")
            for fid in order:
                arr = np.array(Image.open(io.BytesIO(d["frames"][fid])).convert("RGB"))
                w.append_data(arr)
            w.close()
            n_ok += 1
        except Exception as e:
            print(f"  LỖI {vid}: {e}")
            n_skip += 1
    print(f"done. {n_ok} clip ghi/đã có, {n_skip} bỏ → {OUT}")


if __name__ == "__main__":
    main()
