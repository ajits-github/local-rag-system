"""`IngestionWorker.run_forever` never claims more jobs than it has free slots (DIST-006).

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

    def reclaim_stale(self, consumer_name, idle_ms):
        return []

    def claim(self, consumer_name, count, block_ms):
        self.claim_counts.append(count)
        jobs = [SimpleNamespace(redelivered=False, n=self._next + i) for i in range(count)]
        self._next += count
        return jobs


def test_run_forever_claims_only_free_slots_while_jobs_are_running(monkeypatch) -> None:
    """With every slot busy, the loop waits instead of claiming another batch."""
    queue = _QueueStub()
    release = threading.Event()
    worker = IngestionWorker(
        queue,
        lambda payload: {},
        WorkerConfig(consumer_name="w1", max_concurrency=2, claim_block_ms=20, reclaim_interval_s=0),
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
