"""Part 3/EXPERIMENTS: distributed-lock scenario, and why a DB constraint is often better still."""

from __future__ import annotations

from distributed_state_experiment.distributed_lock.demo_lock import (
    RacyDocumentIdTable,
    RedisDistributedLock,
    run_concurrent_get_or_create,
)


def test_unsafe_concurrent_get_or_create_produces_more_than_one_document_id() -> None:
    """Without any coordination, concurrent callers race and disagree on the document_id."""
    table = RacyDocumentIdTable(race_window_seconds=0.05)
    results = run_concurrent_get_or_create(
        table, "security/policy.md", "techfusion", worker_count=8, use_lock=None
    )
    assert len(set(results)) > 1, "the race should have produced more than one distinct document_id"


def test_redis_lock_serializes_the_same_race_to_one_id(fake_redis) -> None:
    """Wrapping the exact same racy method in a Redis `SET NX PX` lock fixes it."""
    table = RacyDocumentIdTable(race_window_seconds=0.05)
    lock = RedisDistributedLock(
        fake_redis, name="doc_ingest:security/policy.md:techfusion", ttl_ms=5000
    )
    results = run_concurrent_get_or_create(
        table, "security/policy.md", "techfusion", worker_count=8, use_lock=lock
    )
    assert (
        len(set(results)) == 1
    ), "the lock should have serialized every caller onto one document_id"


def test_atomic_check_and_set_fixes_the_race_with_no_lock_at_all() -> None:
    """The DB-constraint-shaped fix: one atomic check-and-set needs no external coordination.

    Mirrors this repository's actual, shipped fix for the equivalent real
    race (`PgVectorStore.get_or_create_document_id`'s `INSERT ... ON
    CONFLICT`): correctness here does not depend on every caller
    remembering to acquire a lock first, because there is no window
    between "check" and "set" for a second caller to land in.
    """
    import threading

    table = RacyDocumentIdTable(race_window_seconds=0.05)
    results: list[str] = []
    results_lock = threading.Lock()

    def _call() -> None:
        doc_id = table.get_or_create_atomic("security/policy.md", "techfusion")
        with results_lock:
            results.append(doc_id)

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1


def test_lock_holder_cannot_release_a_lock_it_no_longer_owns(fake_redis) -> None:
    """The token-checked release prevents releasing a lock re-acquired by someone else after TTL expiry."""
    lock_a = RedisDistributedLock(fake_redis, name="doc_ingest:x.md:ds1", ttl_ms=50)
    assert lock_a.acquire(timeout_seconds=1.0)

    import time

    time.sleep(0.1)  # let lock_a's TTL expire

    lock_b = RedisDistributedLock(fake_redis, name="doc_ingest:x.md:ds1", ttl_ms=5000)
    assert lock_b.acquire(timeout_seconds=1.0), "lock_b should acquire the now-expired lock"

    # lock_a's stale release must be a no-op, not a hijack of lock_b's lock.
    assert lock_a.release() is False
    assert fake_redis.get(lock_b.key) is not None, "lock_b's lock must still be held"
    assert lock_b.release() is True


def test_second_acquire_attempt_fails_while_lock_is_held(fake_redis) -> None:
    lock_a = RedisDistributedLock(fake_redis, name="doc_ingest:y.md:ds1", ttl_ms=5000)
    lock_b = RedisDistributedLock(fake_redis, name="doc_ingest:y.md:ds1", ttl_ms=5000)
    assert lock_a.acquire(timeout_seconds=1.0)
    assert lock_b.acquire(timeout_seconds=0.1) is False
    lock_a.release()
    assert lock_b.acquire(timeout_seconds=1.0)
    lock_b.release()
