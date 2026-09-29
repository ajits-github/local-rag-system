"""Query one job's status, or print a summary of queue/dead-letter/backlog depth.

Example
-------
    python -m distributed_state_experiment.cli.job_status --summary
    python -m distributed_state_experiment.cli.job_status --job-id job-...
"""

from __future__ import annotations

import argparse

import redis

from distributed_state_experiment.job_queue.queue import IngestionJobQueue


def main() -> None:
    """Parse args and print either one job's record or a queue-wide summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    parser.add_argument("--stream", default="ingest_jobs:stream")
    parser.add_argument("--group", default="ingest_workers")
    parser.add_argument("--job-id")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    client = redis.Redis.from_url(args.redis_url)
    queue = IngestionJobQueue(client, stream_name=args.stream, group_name=args.group)

    if args.job_id:
        record = queue.store.get(args.job_id)
        if record is None:
            print(f"no such job: {args.job_id}")
            return
        print(record)
        return

    print(
        f"backlog_depth={queue.backlog_depth()} "
        f"stream_xlen={queue.queue_depth()} "
        f"pending={queue.pending_count()} "
        f"dead_letter_depth={queue.dead_letter_depth()} "
        f"active_workers={queue.active_worker_count()}"
    )


if __name__ == "__main__":
    main()
