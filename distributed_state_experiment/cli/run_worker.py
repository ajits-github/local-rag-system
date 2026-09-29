"""Run one ingestion worker replica (Experiments 2-5, 11-12).

By default uses `simulated_ingestion_process_fn` (fast, no `rag`/torch
import, no Postgres needed -- see `job_queue/worker.py`'s module
docstring for why this is what this session's own experiment runs used).
Pass `--real` to instead build the genuine `IngestionPipeline`-backed
`process_fn` against `RAG_CONFIG_PATH`/`DATABASE_URL` (needs the real
app's dependencies importable and a reachable Postgres).

Example
-------
    python -m distributed_state_experiment.cli.run_worker --name worker-1
    # in another terminal:
    python -m distributed_state_experiment.cli.run_worker --name worker-2
"""

from __future__ import annotations

import argparse
import logging

import redis

from distributed_state_experiment.job_queue.queue import IngestionJobQueue
from distributed_state_experiment.job_queue.worker import (
    IngestionWorker,
    WorkerConfig,
    real_ingestion_process_fn,
    simulated_ingestion_process_fn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("run_worker")


def main() -> None:
    """Parse args and run one worker replica until a signal or the queue idles out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6399/0")
    parser.add_argument("--stream", default="ingest_jobs:stream")
    parser.add_argument("--group", default="ingest_workers")
    parser.add_argument("--name", required=True, help="this replica's unique consumer name")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--reclaim-idle-ms", type=int, default=3000, help="visibility timeout")
    parser.add_argument("--base-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=10.0)
    parser.add_argument(
        "--run-until-idle",
        action="store_true",
        help="exit once the queue is drained, instead of running forever",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real IngestionPipeline instead of the simulated process_fn",
    )
    args = parser.parse_args()

    client = redis.Redis.from_url(args.redis_url)
    queue = IngestionJobQueue(client, stream_name=args.stream, group_name=args.group)

    if args.real:
        from rag.config import load_config

        process_fn = real_ingestion_process_fn(load_config())
    else:
        process_fn = simulated_ingestion_process_fn

    worker = IngestionWorker(
        queue,
        process_fn,
        WorkerConfig(
            consumer_name=args.name,
            max_concurrency=args.max_concurrency,
            reclaim_idle_ms=args.reclaim_idle_ms,
            base_backoff_seconds=args.base_backoff_seconds,
            max_backoff_seconds=args.max_backoff_seconds,
        ),
    )
    logger.info("worker %s starting (real=%s)", args.name, args.real)
    if args.run_until_idle:
        processed = worker.run_until_idle(idle_polls=3)
        logger.info("worker %s: drained after processing %d attempts", args.name, processed)
    else:
        worker.install_signal_handlers()
        worker.run_forever()


if __name__ == "__main__":
    main()
