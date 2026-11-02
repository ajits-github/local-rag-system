"""`IngestionWorker.run_forever` never takes more jobs than it has free slots (DIST-006).

The loop used to claim `max_concurrency` jobs on every pass regardless of jobs
still running, so claimed-but-unstarted jobs piled up in the executor, aged
past `reclaim_idle_ms`, and could be reclaimed and run twice by another worker.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from distributed_state_experiment.job_queue.worker import IngestionWorker, WorkerConfig


class _QueueStub:
    """Hands out a job for every requested slot and records each claim's `count`."""

    def __init__(self) -> None:
        self.claim_counts: list[int] = []
        self.reclaim_counts: list[int] = []
        self.renewed: list[list[str]] = []
        self.stale = 0
        self._next = 0

    def heartbeat(self, consumer_name):
        pass

    def active_worker_count(self):
        return 1

    def promote_due_retries(self):
        pass

    def backlog_depth(self):
        return 0

    def dead_letter_depth(self):
        return 0

    def reclaim_stale(self, consumer_name, idle_ms, count=10):
        self.reclaim_counts.append(count)
        jobs = [SimpleNamespace(entry_id=f"stale-{i}", redelivered=True) for i in range(self.stale)]
        self.stale = 0
        return jobs[:count]

    def renew(self, consumer_name, entry_ids):
        self.renewed.append(list(entry_ids))

    def claim(self, consumer_name, count, block_ms):
        self.claim_counts.append(count)
        jobs = [
            SimpleNamespace(entry_id=f"new-{self._next + i}", redelivered=False)
            for i in range(count)
        ]
        self._next += count
        return jobs


def test_run_forever_claims_only_free_slots_while_jobs_are_running(monkeypatch) -> None:
    """With every slot busy, the loop waits instead of claiming another batch."""
    queue = _QueueStub()
    release = threading.Event()
    worker = IngestionWorker(
        queue,
        lambda payload: {},
        WorkerConfig(
            consumer_name="w1", max_concurrency=2, claim_block_ms=20, reclaim_interval_s=0
        ),
    )
    monkeypatch.setattr(worker, "_process_one", lambda job: release.wait(5))

    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    time.sleep(0.3)  # many loop passes while both jobs block
    in_flight_claims = list(queue.claim_counts)
    worker.request_shutdown()
    release.set()
    thread.join(timeout=5)

    assert in_flight_claims == [2]
    assert sum(queue.claim_counts) <= 2 + 2  # at most one refill after release


def _run_saturated(queue, *, max_concurrency=2, reclaim_idle_ms=30_000, seconds=0.3):
    """Run the production loop with every submitted job blocked; return the started-job ids."""
    release = threading.Event()
    submitted: list[str] = []
    worker = IngestionWorker(
        queue,
        lambda payload: {},
        WorkerConfig(
            consumer_name="w1",
            max_concurrency=max_concurrency,
            claim_block_ms=20,
            reclaim_interval_s=0,
            reclaim_idle_ms=reclaim_idle_ms,
        ),
    )

    def process(job):
        submitted.append(job.entry_id)
        release.wait(5)

    worker._process_one = process
    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    time.sleep(seconds)
    during = (list(queue.claim_counts), list(queue.reclaim_counts))
    worker.request_shutdown()
    release.set()
    thread.join(timeout=5)
    return during, submitted


def test_reclaim_takes_no_more_than_the_free_slots():
    """Ten stale entries do not enter a two-slot worker's pool at once (DIST-N01)."""
    queue = _QueueStub()
    queue.stale = 10

    (claim_counts, reclaim_counts), submitted = _run_saturated(queue)

    assert reclaim_counts == [2]
    assert claim_counts == []
    assert sorted(submitted) == ["stale-0", "stale-1"]


def test_reclaimed_jobs_count_against_capacity_for_later_claims():
    """A saturated worker neither claims new jobs nor reclaims more on later passes."""
    queue = _QueueStub()
    queue.stale = 1

    (claim_counts, reclaim_counts), _ = _run_saturated(queue)

    assert reclaim_counts[0] == 2
    assert claim_counts[0] == 1
    assert sum(claim_counts) == 1


def test_running_jobs_are_renewed_so_long_jobs_are_not_reclaimed():
    """A job outliving the visibility timeout must not look crashed to other workers."""
    queue = _QueueStub()

    _run_saturated(queue, reclaim_idle_ms=60, seconds=0.4)

    assert sorted(set(sum(queue.renewed, []))) == ["new-0", "new-1"]


def test_renewed_entry_is_no_longer_stale(fake_redis):
    """Against the real stream commands: a renewed entry is no longer stale."""
    from distributed_state_experiment.job_queue.models import IngestionJobPayload
    from distributed_state_experiment.job_queue.queue import IngestionJobQueue

    queue = IngestionJobQueue(fake_redis)
    queue.submit(IngestionJobPayload(source_path="a.md", dataset_id="ds", idempotency_key="k"))
    [job] = queue.claim("w1", count=1, block_ms=1)
    time.sleep(0.05)

    queue.renew("w1", [job.entry_id])

    assert queue.reclaim_stale("w2", min_idle_ms=30) == []
