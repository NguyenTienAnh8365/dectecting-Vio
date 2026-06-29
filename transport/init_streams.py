"""Khởi tạo consumer group (chạy 1 lần). Đọc từ cuối stream ($) — bỏ frame cũ tồn đọng."""
from __future__ import annotations

import redis
from redis.exceptions import ResponseError

from . import config


def ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    try:
        r.xgroup_create(name=stream, groupname=group, id="$", mkstream=True)
        print(f"  created group {group} on {stream}")
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"  group {group} on {stream} already exists")
        else:
            raise


def main() -> None:
    r = redis.from_url(config.REDIS_URL)
    print(f"Redis: {config.REDIS_URL}  ping={r.ping()}")
    ensure_group(r, config.REQ_STREAM, config.CG_GATE)
    ensure_group(r, config.GATED_STREAM, config.CG_CLS)
    ensure_group(r, config.RES_STREAM, config.CG_DASH)
    print("init done.")


if __name__ == "__main__":
    main()
