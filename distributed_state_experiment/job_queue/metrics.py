"""Prometheus metrics for the distributed ingestion job queue and workers.

Mirrors `src/rag/observability/metrics.py`'s own pattern exactly (a
dedicated `CollectorRegistry`, not `prometheus_client`'s process-wide
default, so re-importing this module across `pytest` test collection
never raises "Duplicated timeseries"; every label drawn from a small,
fixed vocabulary). Kept inside `distributed_state_experiment/` rather
than added to `rag.observability` itself, since this experiment's queue
metrics describe infrastructure (`distributed_state_experiment/job_queue`)
that is not part of the production `rag.api` process.

`job_status` is the one label used across several metrics, and it is
always one of `JobStatus`'s fixed enum values -- never a raw job id,
dataset id, or file path.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

_LATENCY_BUCKETS_SECONDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
_ATTEMPT_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10)

JOBS_SUBMITTED_TOTAL = Counter(
    "ingest_jobs_submitted_total",
    "Total ingestion jobs submitted (excludes idempotent no-op resubmissions).",
    registry=REGISTRY,
)
JOBS_COMPLETED_TOTAL = Counter(
    "ingest_jobs_completed_total",
    "Total ingestion jobs that reached COMPLETED.",
    registry=REGISTRY,
)
JOBS_FAILED_TOTAL = Counter(
    "ingest_jobs_failed_total",
    "Total individual processing attempts that raised (whether retried or dead-lettered next).",
    registry=REGISTRY,
)
JOBS_RETRIED_TOTAL = Counter(
    "ingest_jobs_retried_total",
    "Total retry schedules created (one per failed attempt that still had budget left).",
    registry=REGISTRY,
)
JOBS_DEAD_LETTERED_TOTAL = Counter(
    "ingest_jobs_dead_lettered_total",
    "Total jobs that exhausted their retry budget and moved to the dead-letter stream.",
    registry=REGISTRY,
)
JOBS_DUPLICATE_DELIVERIES_TOTAL = Counter(
    "ingest_jobs_duplicate_deliveries_total",
    "Total claims that were redeliveries (via reclaim_stale) of an already-claimed entry.",
    registry=REGISTRY,
)

JOB_PROCESSING_LATENCY_SECONDS = Histogram(
    "ingest_job_processing_latency_seconds",
    "Wall-clock time for one processing attempt (claim to ack/fail), by outcome.",
    ["outcome"],
    buckets=_LATENCY_BUCKETS_SECONDS,
    registry=REGISTRY,
)
JOB_ATTEMPTS_AT_COMPLETION = Histogram(
    "ingest_job_attempts_at_completion",
    "Number of attempts a job needed before reaching a terminal state (COMPLETED or DEAD_LETTER).",
    ["terminal_status"],
    buckets=_ATTEMPT_BUCKETS,
    registry=REGISTRY,
)

QUEUE_DEPTH = Gauge(
    "ingest_queue_depth",
    "Current main-stream length (jobs not yet removed from the work stream).",
    registry=REGISTRY,
)
DEAD_LETTER_DEPTH = Gauge(
    "ingest_dead_letter_depth",
    "Current dead-letter stream length.",
    registry=REGISTRY,
)
RETRY_QUEUE_DEPTH = Gauge(
    "ingest_retry_queue_depth",
    "Number of jobs currently waiting on their backoff timer before re-enqueue.",
    registry=REGISTRY,
)
ACTIVE_WORKERS = Gauge(
    "ingest_active_workers",
    "Number of worker processes currently registered as running (heartbeat-based).",
    registry=REGISTRY,
)


def observe_job_processed(outcome: str, duration_seconds: float) -> None:
    """Record one processing attempt's outcome and latency.

    Parameters
    ----------
    outcome : {"completed", "failed"}
        The fixed, bounded label value for this attempt.
    duration_seconds : float
        Wall-clock duration of the attempt.
    """
    JOB_PROCESSING_LATENCY_SECONDS.labels(outcome=outcome).observe(duration_seconds)
    if outcome == "completed":
        JOBS_COMPLETED_TOTAL.inc()
    else:
        JOBS_FAILED_TOTAL.inc()


def observe_terminal(status: str, attempts: int) -> None:
    """Record a job reaching a terminal state, and how many attempts it took.

    Parameters
    ----------
    status : {"completed", "dead_letter"}
        Fixed, bounded label value.
    attempts : int
        Total attempts made before reaching this terminal state.
    """
    JOB_ATTEMPTS_AT_COMPLETION.labels(terminal_status=status).observe(attempts)
    if status == "dead_letter":
        JOBS_DEAD_LETTERED_TOTAL.inc()
