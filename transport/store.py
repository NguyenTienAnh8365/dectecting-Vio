"""Lưu bằng chứng ra disk local (snapshot JPG + clip MP4) — dashboard serve trực tiếp.

Demo dùng disk cho gọn (1 server, vài camera). Muốn durable/đa máy thì thay bằng
ImageStore MinIO của Remote Sensing — interface put tương tự, đổi key→object.

Encode chạy trên thread pool để KHÔNG chặn vòng infer của classifier (off hot-path).
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from . import config


class MediaStore:
    def __init__(self, root: str | None = None, max_workers: int = 3):
        self.root = root or config.MEDIA_DIR
        os.makedirs(self.root, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="media")

    @staticmethod
    def _key(camera_id: str, ts: float, kind: str, ext: str) -> str:
        day = datetime.fromtimestamp(ts).strftime("%Y%m%d")
        stamp = datetime.fromtimestamp(ts).strftime("%H%M%S")
        ms = int((ts % 1) * 1000)
        return f"{kind}/{day}/{camera_id}/{stamp}_{ms:03d}.{ext}"

    def save_snapshot_async(self, camera_id: str, ts: float, bgr: np.ndarray, quality: int = 85) -> str:
        """Lưu 1 ảnh snapshot (đã vẽ box nếu muốn). Trả key tương đối (dashboard ghép /media/<key>)."""
        key = self._key(camera_id, ts, "snapshot", "jpg")
        path = os.path.join(self.root, key)
        img = bgr.copy()  # caller có thể sửa frame sau khi gọi

        def _do():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

        self._pool.submit(_do)
        return key

    def save_clip_async(self, camera_id: str, ts: float, frames: list[np.ndarray], fps: float = 12.0) -> str:
        """Ghi clip MP4 từ list frame BGR (cửa sổ bằng chứng). Trả key tương đối."""
        key = self._key(camera_id, ts, "clip", "mp4")
        path = os.path.join(self.root, key)
        clip = [f.copy() for f in frames]
        if not clip:
            return ""

        def _do():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            h, w = clip[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not vw.isOpened():
                return
            for fr in clip:
                if fr.shape[:2] != (h, w):
                    fr = cv2.resize(fr, (w, h))
                vw.write(fr)
            vw.release()

        self._pool.submit(_do)
        return key

    def cleanup(self, ttl_days: int | None = None) -> int:
        """Xóa file cũ hơn TTL (gọi định kỳ từ dashboard). Trả số file đã xóa."""
        ttl = (ttl_days if ttl_days is not None else config.MEDIA_TTL_DAYS) * 86400
        if ttl <= 0:
            return 0
        cutoff = time.time() - ttl
        removed = 0
        for dirpath, _, files in os.walk(self.root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        removed += 1
                except OSError:
                    pass
        return removed

    def close(self) -> None:
        self._pool.shutdown(wait=True)
