"""Producer/consumer, retry, dead-letter, idempotency, and crash-recovery tests.

All against `fakeredis` (Streams are natively supported; no Lua needed
here), so this whole file runs in-process with no live Redis required --
mirrors experiments 1, 6-10 from `README.md`.
"""

from __future__ import annotations

import time

from distributed_state_experiment.job_queue.models import IngestionJobPayload, JobStatus
from distributed_state_experiment.job_queue.queue import IngestionJobQueue, QueueFullError
from distributed_state_experiment.job_queue.worker import (
    IngestionWorker,
    PoisonJobError,
    WorkerConfig,
    simulated_ingestion_process_fn,
)


def _payload(
    path: str = "docs/a.md", dataset_id: str = "ds1", poison: bool = False
) -> IngestionJobPayload:
    return IngestionJobPayload.build(
        path, dataset_id, content=f"content-of-{path}".encode(), poison=poison
    )


def test_submit_then_claim_then_complete(fake_redis) -> None:
    queue = IngestionJobQueue(fake_redis, stream_name="s1", group_name="g1")
    record = queue.submit(_payload())
    assert record.status == JobStatus.SUBMITTED
    assert queue.queue_depth() == 1

    claimed = queue.claim("worker-1")
    assert len(claimed) == 1
    assert claimed[0].record.status == JobStatus.PROCESSING
    assert claimed[0].record.attempts == 1

    queue.complete(claimed[0], {"chunks_embedded": 3})
    stored = queue.store.get(record.job_id)
    assert stored.status == JobStatus.COMPLETED
    assert queue.pending_count() == 0


def test_submitting_the_same_payload_twice_returns_the_same_job(fake_redis) -> None:
    """Experiment 6/7: duplicate submission is deduplicated at the queue's own entry point."""
    queue = IngestionJobQueue(fake_redis, stream_name="s2", group_name="g2")
    payload = _payload()
    first = queue.submit(payload)
    second = queue.submit(payload)
    assert first.job_id == second.job_id
    assert queue.queue_depth() == 1, "the second submission must not add a second stream entry"


def test_processing_the_same_job_twice_does_not_double_complete_or_error(fake_redis) -> None:
    """Experiment 7: idempotent *processing*, not just idempotent submission.

    Simulates at-least-once delivery by claiming, then manually re-adding
    the exact same job_id to the stream (as `reclaim_stale`/a crashed
    worker's redelivery would) and processing it again -- the worker's
    `process_fn` runs twice, but that mirrors this repository's real
    idempotency guarantee: `IngestionPipeline`'s checksum-gated
    `replace_document_chunks` makes a second identical ingest a no-op, not
    a duplicate insert. Here, the stand-in `simulated_ingestion_process_fn`
    is itself side-effect-free (just returns a result dict), so this test
    asserts the *queue*-level property that matters: both attempts
    complete the same job_id cleanly, with `attempts=2` recorded, rather
    than crashing or creating a second job record.
    """
    queue = IngestionJobQueue(fake_redis, stream_name="s3", group_name="g3")
    payload = _payload()
    record = queue.submit(payload)

    first_claim = queue.claim("worker-1")[0]
    queue.complete(first_claim, {"chunks_embedded": 3})

    # Simulate a duplicate delivery of the *same* job_id (e.g. a message
    # that was reclaimed after appearing to be stale, but whose original
    # worker was actually still alive and also completes it).
    fake_redis.xadd("s3", {"job_id": record.job_id})
    second_claim = queue.claim("worker-2")[0]
    assert second_claim.record.attempts == 2
    queue.complete(second_claim, {"chunks_embedded": 3})

    final = queue.store.get(record.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.attempts == 2


def test_failing_job_schedules_a_retry_with_backoff(fake_redis) -> None:
    """Experiment 8/9: a failing job is retried, not immediately re-delivered."""
    queue = IngestionJobQueue(fake_redis, stream_name="s4", group_name="g4")
    payload = _payload(poison=True)
    record = queue.submit(payload, max_attempts=3)

    claimed = queue.claim("worker-1")[0]
    queue.fail(claimed, PoisonJobError("boom"), base_backoff_seconds=1.0, max_backoff_seconds=60.0)

    stored = queue.store.get(record.job_id)
    assert stored.status == JobStatus.RETRY_SCHEDULED
    assert stored.attempts == 1
    assert "PoisonJobError" in stored.last_error

    # Not due yet: promoting "now" should not re-enqueue it. The job is
    # still outstanding (backlog_depth), just not currently claimable.
    assert queue.promote_due_retries(now=time.time()) == 0
    assert queue.backlog_depth() == 1

    # Once its backoff has elapsed, promotion puts it back on the main stream.
    future = time.time() + 100
    promoted = queue.promote_due_retries(now=future)
    assert promoted == 1
    assert queue.backlog_depth() == 1  # still outstanding -- promotion isn't completion
    # XLEN grew by one: the retry is a brand new stream entry, and the
    # original (acked) entry is still physically present in the stream --
    # XACK never deletes it. This is the real Streams gotcha queue_depth()
    # documents: XLEN measures "entries ever added," not "work remaining."
    assert queue.queue_depth() == 2


def test_exhausting_retries_moves_the_job_to_the_dead_letter_stream(fake_redis) -> None:
    """Experiment 10: exhaust the retry budget, land in the dead-letter stream."""
    queue = IngestionJobQueue(fake_redis, stream_name="s5", group_name="g5")
    payload = _payload(poison=True)
    record = queue.submit(payload, max_attempts=2)

    for _ in range(2):
        claimed = queue.claim("worker-1")[0]
        queue.fail(claimed, PoisonJobError("boom"))
        queue.promote_due_retries(now=time.time() + 1000)

    stored = queue.store.get(record.job_id)
    assert stored.status == JobStatus.DEAD_LETTER
    assert stored.attempts == 2
    assert queue.dead_letter_depth() == 1
    assert queue.backlog_depth() == 0


def test_reclaim_stale_recovers_a_job_from_a_crashed_worker(fake_redis) -> None:
    """Experiment 4/5: a worker claims a job and "crashes" (never acks); another worker recovers it."""
    queue = IngestionJobQueue(fake_redis, stream_name="s6", group_name="g6")
    queue.submit(_payload())

    crashed_claim = queue.claim("worker-doomed")
    assert len(crashed_claim) == 1
    assert queue.pending_count() == 1  # claimed but never acked -- the crashed worker's PEL entry

    # A real visibility timeout waits much longer than this; a short real
    # sleep here just guarantees the entry's idle time is measurably
    # nonzero (Redis's `idle` filter is a real millisecond-resolution
    # elapsed-time check, not a logical clock), so min_idle_ms=0 reliably
    # matches it instead of flaking on sub-millisecond timing.
    time.sleep(0.02)
    reclaimed = queue.reclaim_stale("worker-rescuer", min_idle_ms=0)
    assert len(reclaimed) == 1
    assert reclaimed[0].redelivered is True
    assert reclaimed[0].record.attempts == 2  # once for the doomed claim, once for the reclaim

    queue.complete(reclaimed[0], {"chunks_embedded": 1})
    assert queue.pending_count() == 0


def test_backpressure_rejects_submission_over_max_queue_depth(fake_redis) -> None:
    queue = IngestionJobQueue(fake_redis, stream_name="s7", group_name="g7")
    queue.submit(_payload("a.md"), max_queue_depth=2)
    queue.submit(_payload("b.md"), max_queue_depth=2)
    try:
        queue.submit(_payload("c.md"), max_queue_depth=2)
        raised = False
    except QueueFullError:
        raised = True
    assert raised


def test_worker_run_until_idle_processes_a_batch_and_dead_letters_a_poison_job(fake_redis) -> None:
    """An end-to-end slice of experiments 1/8/9/10 through the real `IngestionWorker` loop."""
    queue = IngestionJobQueue(fake_redis, stream_name="s8", group_name="g8")
    for i in range(5):
        queue.submit(_payload(f"doc-{i}.md"))
    poison_record = queue.submit(_payload("poison.md", poison=True), max_attempts=1)

    worker = IngestionWorker(
        queue,
        simulated_ingestion_process_fn,
        WorkerConfig(consumer_name="worker-1", max_concurrency=2, reclaim_interval_s=0),
    )
    # max_attempts=1 means the poison job's very first failure already
    # exhausts its retry budget, so one drain is enough to dead-letter it
    # (no backoff/promotion cycle needed for this particular assertion).
    worker.run_until_idle(idle_polls=2)

    for i in range(5):
        record = queue.store.find_by_idempotency_key(_payload(f"doc-{i}.md").idempotency_key)
        assert record.status == JobStatus.COMPLETED

    poison_final = queue.store.get(poison_record.job_id)
    assert poison_final.status == JobStatus.DEAD_LETTER
    assert queue.dead_letter_depth() == 1
