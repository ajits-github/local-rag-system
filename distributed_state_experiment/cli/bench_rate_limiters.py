"""Part 4 benchmark: single-replica in-memory vs 3-replica in-memory vs 3-replica Redis-backed.

Each "replica" is a genuinely separate OS process (via `subprocess.Popen`,
running `rate_limit_bench_worker.py`), each firing 100 requests at the
same intended-100/minute tenant bucket. Reports, for each of the three
configurations: total requests allowed globally (correctness -- the
intended limit is 100 in every configuration) and total wall-clock time
(latency/throughput cost of the fix).

Usage
-----
    python -m distributed_state_experiment.cli.bench_rate_limiters --redis-url redis://127.0.0.1:6399/0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid

PYTHON = sys.executable
MODULE = "distributed_state_experiment.cli.rate_limit_bench_worker"


def _run_replicas(
    backend: str, replica_count: int, requests_per_replica: int, bucket: str, redis_url: str
) -> tuple[int, float, list[int]]:
    """Launch `replica_count` concurrent worker processes and aggregate their results.

    Returns
    -------
    tuple[int, float, list[int]]
        Total allowed across all replicas, total wall-clock seconds for
        the whole batch, and each replica's individually-allowed count
        (to show the per-replica split, matching the task's own
        "Pod A: counter = 70 / Pod B: counter = 70" example shape).
    """
    cmd_base = [
        PYTHON,
        "-m",
        MODULE,
        "--backend",
        backend,
        "--redis-url",
        redis_url,
        "--bucket",
        bucket,
        "--limit",
        "100",
        "--count",
        str(requests_per_replica),
    ]
    start = time.perf_counter()
    procs = [
        subprocess.Popen(cmd_base, stdout=subprocess.PIPE, text=True) for _ in range(replica_count)
    ]
    outputs = [p.communicate()[0].strip() for p in procs]
    elapsed = time.perf_counter() - start

    per_replica_allowed = []
    for line in outputs:
        allowed_str, _latency_ms = line.split("\t")
        per_replica_allowed.append(int(allowed_str))
    return sum(per_replica_allowed), elapsed, per_replica_allowed


def main() -> None:
    """Run the three configurations and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    parser.add_argument("--requests-per-replica", type=int, default=100)
    args = parser.parse_args()
    run_id = uuid.uuid4().hex[:8]

    print(
        f"{'configuration':<38}{'total_allowed':>15}{'intended_limit':>16}{'wall_clock_s':>14}{'per_replica':>20}"
    )

    # 1. Single replica, in-memory (the correct, single-instance case).
    allowed, elapsed, per_replica = _run_replicas(
        "memory",
        1,
        args.requests_per_replica,
        f"tenant:bench:{run_id}:single-memory",
        args.redis_url,
    )
    print(
        f"{'single replica + in-memory':<38}{allowed:>15}{100:>16}{elapsed:>14.3f}{str(per_replica):>20}"
    )

    # 2. Three replicas, in-memory (the bug: each replica's own 100-cap adds up).
    allowed, elapsed, per_replica = _run_replicas(
        "memory", 3, args.requests_per_replica, f"tenant:bench:{run_id}:3-memory", args.redis_url
    )
    print(
        f"{'3 replicas + in-memory':<38}{allowed:>15}{100:>16}{elapsed:>14.3f}{str(per_replica):>20}"
    )

    # 3. Three replicas, Redis-backed (the fix: one shared counter).
    allowed, elapsed, per_replica = _run_replicas(
        "redis", 3, args.requests_per_replica, f"tenant:bench:{run_id}:3-redis", args.redis_url
    )
    print(
        f"{'3 replicas + Redis-backed':<38}{allowed:>15}{100:>16}{elapsed:>14.3f}{str(per_replica):>20}"
    )


if __name__ == "__main__":
    main()
