"""One benchmark worker process: fires `--count` requests at one bucket, reports allowed/latency.

Used by `bench_rate_limiters.py` to drive Part 4's three-way comparison
as genuinely separate OS processes (not threads in one process), since
"process-local state" is only a faithful demonstration when the state
really is process-local.
"""

from __future__ import annotations

import argparse
import time

import redis

from distributed_state_experiment.rate_limiter.in_memory import InMemoryFixedWindowLimiter
from distributed_state_experiment.rate_limiter.redis_fixed_window_limiter import (
    RedisFixedWindowLimiter,
)


def main() -> None:
    """Parse args, fire `--count` requests at `--bucket`, print `allowed<TAB>latency_ms`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["memory", "redis"], required=True)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    parser.add_argument("--bucket", default="tenant:bench")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    start = time.perf_counter()
    if args.backend == "memory":
        limiter = InMemoryFixedWindowLimiter(limit=args.limit, window_seconds=60)
        allowed = sum(1 for _ in range(args.count) if limiter.hit(args.bucket)[0])
    else:
        client = redis.Redis.from_url(args.redis_url)
        rlimiter = RedisFixedWindowLimiter(client, window_seconds=60, key_prefix="bench_rl")
        allowed = sum(
            1 for _ in range(args.count) if rlimiter.allow(args.bucket, args.limit).allowed
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{allowed}\t{elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
