"""Submit N ingestion jobs to a running queue (Experiment 1 and friends).

Example
-------
    python -m distributed_state_experiment.cli.submit_jobs --count 20
    python -m distributed_state_experiment.cli.submit_jobs --count 1 --poison
"""

from __future__ import annotations

import argparse

import redis

from distributed_state_experiment.job_queue.models import IngestionJobPayload
from distributed_state_experiment.job_queue.queue import IngestionJobQueue


def main() -> None:
    """Parse args, submit `--count` jobs, and print each resulting job id."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    parser.add_argument("--stream", default="ingest_jobs:stream")
    parser.add_argument("--group", default="ingest_workers")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--dataset-id", default="distributed_state_demo")
    parser.add_argument(
        "--poison", action="store_true", help="submit poison (always-failing) jobs instead"
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--prefix", default="doc", help="synthetic source-path prefix")
    args = parser.parse_args()

    client = redis.Redis.from_url(args.redis_url)
    queue = IngestionJobQueue(client, stream_name=args.stream, group_name=args.group)

    for i in range(args.count):
        content = f"synthetic content for {args.prefix}-{i}, run marker {id(args)}".encode()
        payload = IngestionJobPayload.build(
            f"{args.prefix}-{i}.md", args.dataset_id, content=content, poison=args.poison
        )
        record = queue.submit(payload, max_attempts=args.max_attempts)
        print(f"{record.job_id}\t{record.status.value}\t{payload.idempotency_key}")

    print(
        f"# submitted={args.count} backlog_depth={queue.backlog_depth()} queue_xlen={queue.queue_depth()}"
    )


if __name__ == "__main__":
    main()
