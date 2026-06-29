"""Đóng/mở gói message envelope (fork RS/transport/envelope.py).

Redis stream value binary-safe ⇒ nhét thẳng bytes JPEG, KHÔNG base64.
`meta`/`result` gói msgpack. 3 loại entry: req (frame thô), gated (frame + person),
res (kết quả phân loại clip).
"""
from __future__ import annotations

from typing import Any

import msgpack


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack(blob: bytes) -> Any:
    return msgpack.unpackb(blob, raw=False)


# --- REQ: grabber → gate ----------------------------------------------------
def build_req(frame_id: str, camera_id: str, ts: float, seq: int, img_bytes: bytes) -> dict[bytes, bytes]:
    return {
        b"frame_id": frame_id.encode(),
        b"camera_id": camera_id.encode(),
        b"ts": repr(ts).encode(),
        b"seq": str(seq).encode(),
        b"img": img_bytes,
    }


def parse_req(fields: dict[bytes, bytes]) -> dict[str, Any]:
    return {
        "frame_id": fields[b"frame_id"].decode(),
        "camera_id": fields[b"camera_id"].decode(),
        "ts": float(fields[b"ts"]),
        "seq": int(fields[b"seq"]),
        "img": fields[b"img"],
    }


# --- GATED: gate → classifier (frame + số người + bbox) ---------------------
def build_gated(req: dict[str, Any], num_persons: int, boxes: list, img_bytes: bytes) -> dict[bytes, bytes]:
    return {
        b"frame_id": req["frame_id"].encode(),
        b"camera_id": req["camera_id"].encode(),
        b"ts": repr(req["ts"]).encode(),
        b"seq": str(req["seq"]).encode(),
        b"persons": pack({"n": num_persons, "boxes": boxes}),
        b"img": img_bytes,
    }


def parse_gated(fields: dict[bytes, bytes]) -> dict[str, Any]:
    persons = unpack(fields[b"persons"])
    return {
        "frame_id": fields[b"frame_id"].decode(),
        "camera_id": fields[b"camera_id"].decode(),
        "ts": float(fields[b"ts"]),
        "seq": int(fields[b"seq"]),
        "num_persons": persons["n"],
        "boxes": persons["boxes"],
        "img": fields[b"img"],
    }


# --- RES: classifier → dashboard --------------------------------------------
def build_res(camera_id: str, result: Any) -> dict[bytes, bytes]:
    """`result` (msgpack): prob, num_persons, ts, seq, snapshot_key, clip_key, model, modelVersion."""
    return {
        b"camera_id": camera_id.encode(),
        b"result": pack(result),
    }


def parse_res(fields: dict[bytes, bytes]) -> dict[str, Any]:
    return {
        "camera_id": fields[b"camera_id"].decode(),
        "result": unpack(fields[b"result"]),
    }
