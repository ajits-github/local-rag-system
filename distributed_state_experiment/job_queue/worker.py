"""Worker replica: claims jobs, runs the ingestion step, acks/retries/dead-letters.

`IngestionWorker` is deliberately decoupled from *what* processing means
via the injected `process_fn` callable, for one honest, documented
reason: this environment's `pip install`-ed packages take several
minutes just to import `sentence_transformers`/`torch` the first time
(confirmed directly: `import rag.factory` alone took over three minutes
on this machine), which makes iterating on 12 short experiments
against the *real* embedding/DB path impractical inside one working
session. `real_ingestion_process_fn` is the genuine integration point
(reuses `rag.factory`/`rag.ingestion.pipeline.IngestionPipeline`/
`rag.vectorstore`/`rag.embedders` exactly as `rag.api.deps` does, no
reimplemented chunking or embedding); `simulated_ingestion_process_fn` is
what this session's own experiment traces in `README.md` actually ran
against, with that substitution called out explicitly wherever it
matters (README's "What was executed vs. simulated" section). The queue/
worker mechanics themselves (claim, ack, retry, backoff, dead-letter,
idempotency, bounded concurrency, graceful shutdown) are identical either
way. Swapping `process_fn` never touches this file.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import metrics
from .models import IngestionJobPayload
from .queue import ClaimedJob, IngestionJobQueue

logger = logging.getLogger("distributed_state_experiment.worker")

ProcessFn = Callable[[IngestionJobPayload], dict]


class PoisonJobError(RuntimeError):
    """Raised by a `process_fn` for a payload with `poison=True`."""


def simulated_ingestion_process_fn(payload: IngestionJobPayload) -> dict:
    """A fast, dependency-free stand-in for "chunk, embed, store."

    Sleeps briefly (simulating real embedding-model latency) and returns
    a small result dict shaped like what the real pipeline would report.
    Always raises `PoisonJobError` for a `poison=True` payload, regardless
    of whether `source_path` exists. This is what experiments 8-10
    (deliberately failing job, retries/backoff, dead-letter) submit.

    Parameters
    ----------
    payload : IngestionJobPayload

    Returns
    -------
    dict
        `{"chunks_embedded": int, "source": str, "simulated": True}`.

    Raises
    ------
    PoisonJobError
        If `payload.poison` is `True`.
    """
    time.sleep(0.05 + payload.slow_seconds)
    if payload.poison:
        raise PoisonJobError(f"poison job: {payload.source_path}")
    return {"chunks_embedded": 3, "source": payload.source_path, "simulated": True}


def real_ingestion_process_fn(config) -> ProcessFn:  # noqa: ANN001 - rag.config.AppConfig, imported lazily
    """Build a `process_fn` backed by the real `IngestionPipeline`.

    Imports `rag.factory`/`rag.ingestion.pipeline` lazily, inside this
    function, precisely so that importing `distributed_state_experiment.job_queue.worker`
    itself never pays the multi-minute `sentence_transformers`/`torch`
    import cost unless a caller actually asks for the real backend.

    Reuses `IngestionPipeline.ingest_file` exactly as `rag.api.routers.ingest`
    and the CLI do. No reimplemented chunking, embedding, or persistence.
    The already-existing checksum-gated idempotency in
    `PgVectorStore.replace_document_chunks` (one atomic transaction; see
    the root `CLAUDE.md`'s "joint-investigation backlog" section) is what
    makes processing the same job twice genuinely safe, not anything this
    module adds.

    Parameters
    ----------
    config : rag.config.AppConfig
        Already-loaded application config (same object `rag.api.deps.get_config()`
        would build).

    Returns
    -------
    ProcessFn
    """
    from rag.factory import build_embedder, build_vectorstore
    from rag.ingestion.pipeline import IngestionPipeline

    vectorstore = build_vectorstore(config)
    embedder = build_embedder(config)
    pipeline = IngestionPipeline(config, vectorstore=vectorstore, embedder=embedder)

    def _process(payload: IngestionJobPayload) -> dict:
        if payload.poison:
            raise PoisonJobError(f"poison job: {payload.source_path}")
        stats = pipeline.ingest_file(payload.source_path, payload.dataset_id)
        return {"source": payload.source_path, "simulated": False, "stats": str(stats)}

    return _process


@dataclass
class WorkerConfig:
    """Tunables for one `IngestionWorker` instance.

    Attributes
    ----------
    consumer_name : str
        This replica's unique consumer-group identity (e.g. `"worker-1"`).
    max_concurrency : int
        Maximum jobs this single worker process processes at once
        (`ThreadPoolExecutor` bound). The "bounded concurrency" the
        spec asks for. A worker never claims more than
        `max_concurrency` un-acked jobs at a time.
    claim_block_ms : int
        How long one `XREADGROUP` call blocks waiting for new entries.
    reclaim_idle_ms : int
        Visibility timeout: an entry pending (claimed, unacked) for at
        least this long is eligible for `reclaim_stale()`. Models a
        worker that claimed a job and then crashed or hung.
    reclaim_interval_s : float
        How often the main loop checks for stale entries to reclaim.
    base_backoff_seconds, max_backoff_seconds : float
        Passed through to `IngestionJobQueue.fail`.
    """

    consumer_name: str
    max_concurrency: int = 4
    claim_block_ms: int = 1000
    reclaim_idle_ms: int = 30_000
    reclaim_interval_s: float = 5.0
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0


class IngestionWorker:
    """One worker replica processing jobs from a shared `IngestionJobQueue`.

    Parameters
    ----------
    queue : IngestionJobQueue
    process_fn : ProcessFn
        Does the actual "chunk, embed, store" work for one payload and
        returns a result dict, or raises on failure. See
        `simulated_ingestion_process_fn`/`real_ingestion_process_fn`.
    config : WorkerConfig
    """

    def __init__(
        self, queue: IngestionJobQueue, process_fn: ProcessFn, config: WorkerConfig
    ) -> None:
        self.queue = queue
        self.process_fn = process_fn
        self.config = config
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrency)
        self._last_reclaim = 0.0

    def request_shutdown(self, *_args: object) -> None:
        """Signal handler / manual trigger: stop claiming new work after the current batch."""
        logger.info("worker %s: shutdown requested", self.config.consumer_name)
        self._shutdown.set()

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that call `request_shutdown` (graceful shutdown).

        Only meaningful when running as the process's main thread; the
        test suite and `bench_limiters.py`-style in-process usage call
        `request_shutdown()`/`stop()` directly instead.
        """
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)

    def run_forever(self) -> None:
        """Main loop: promote due retries, reclaim stale entries, claim and process, until shutdown.

        On shutdown, stops claiming *new* work immediately but waits for
        any jobs already dispatched to the thread pool to finish (graceful
        drain) before returning. A job is never abandoned mid-processing
        just because shutdown was requested; it is only ever abandoned by
        a hard process kill (`kill -9` / `taskkill /F`), which is exactly
        what experiment 4 uses to prove `reclaim_stale()` recovers it.
        """
        futures = []
        while not self._shutdown.is_set():
            self.queue.heartbeat(self.config.consumer_name)
            metrics.ACTIVE_WORKERS.set(self.queue.active_worker_count())
            self._maybe_reclaim()
            self.queue.promote_due_retries()
            metrics.QUEUE_DEPTH.set(self.queue.backlog_depth())
            metrics.DEAD_LETTER_DEPTH.set(self.queue.dead_letter_depth())
            claimed = self.queue.claim(
                self.config.consumer_name,
                count=self.config.max_concurrency,
                block_ms=self.config.claim_block_ms,
            )
            for job in claimed:
                if job.redelivered:
                    metrics.JOBS_DUPLICATE_DELIVERIES_TOTAL.inc()
                futures.append(self._executor.submit(self._process_one, job))
            futures = [f for f in futures if not f.done()]
        self._executor.shutdown(wait=True)
        logger.info("worker %s: drained, exiting", self.config.consumer_name)

    def run_until_idle(self, idle_polls: int = 2) -> int:
        """Process until the queue and retry backlog have been empty for `idle_polls` checks.

        A finite-duration alternative to `run_forever()` for scripted
        experiments/tests, so a demo doesn't need a real signal to stop a
        worker once all submitted work is done.

        Parameters
        ----------
        idle_polls : int, optional
            Consecutive empty claims required before returning.

        Returns
        -------
        int
            Total jobs processed (completed + failed attempts) in this call.
        """
        processed = 0
        empty_streak = 0
        while empty_streak < idle_polls:
            self._maybe_reclaim()
            self.queue.promote_due_retries()
            claimed = self.queue.claim(
                self.config.consumer_name, count=self.config.max_concurrency, block_ms=200
            )
            if not claimed:
                empty_streak += 1
                continue
            empty_streak = 0
            futures = [self._executor.submit(self._process_one, job) for job in claimed]
            for f in futures:
                f.result()
            processed += len(claimed)
        return processed

    def _maybe_reclaim(self) -> None:
        """Run `reclaim_stale()` at most once per `reclaim_interval_s`."""
        now = time.monotonic()
        if now - self._last_reclaim < self.config.reclaim_interval_s:
            return
        self._last_reclaim = now
        reclaimed = self.queue.reclaim_stale(self.config.consumer_name, self.config.reclaim_idle_ms)
        for job in reclaimed:
            metrics.JOBS_DUPLICATE_DELIVERIES_TOTAL.inc()
            self._executor.submit(self._process_one, job)

    def _process_one(self, job: ClaimedJob) -> None:
        """Run `process_fn` for one claimed job and ack/fail it, recording metrics."""
        start = time.monotonic()
        try:
            result = self.process_fn(job.payload)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: any processing failure retries
            duration = time.monotonic() - start
            metrics.observe_job_processed("failed", duration)
            self.queue.fail(
                job, exc, self.config.base_backoff_seconds, self.config.max_backoff_seconds
            )
            if job.record.attempts >= job.record.max_attempts:
                metrics.observe_terminal("dead_letter", job.record.attempts)
            else:
                metrics.JOBS_RETRIED_TOTAL.inc()
            logger.warning(
                "worker %s: job %s failed (attempt %d/%d): %s",
                self.config.consumer_name,
                job.record.job_id,
                job.record.attempts,
                job.record.max_attempts,
                exc,
            )
            return

        duration = time.monotonic() - start
        metrics.observe_job_processed("completed", duration)
        self.queue.complete(job, result)
        metrics.observe_terminal("completed", job.record.attempts)
