"""Race two "workers" ingesting the same logical document, three ways.

The race being reproduced
--------------------------
`PgVectorStore.get_or_create_document_id(source, dataset_id)` used to be
(before the joint-investigation backlog's DATA-1/identity fixes, see the
root `CLAUDE.md`) a racy read-then-write:

    row = SELECT document_id FROM documents WHERE source=%s AND dataset_id=%s
    if row is None:
        INSERT INTO documents (source, dataset_id, document_id) VALUES (...)
    return row.document_id or the newly inserted id

Two concurrent ingestion workers processing the same `(source,
dataset_id)` pair (e.g. two replicas of an ingestion worker fleet both
picking up a resubmitted job for the same file, or a directory walk and
an API upload landing on the same path at the same moment) can both run
the `SELECT`, both see no existing row, and both `INSERT`. That produces
either a duplicate-document-id bug or a unique-constraint crash, depending on
whether a constraint existed at all.

`RacyDocumentIdTable` below reproduces exactly that shape (in-process,
no real Postgres needed) so this module can demonstrate and test the fix
without touching the real database.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import redis


@dataclass
class RacyDocumentIdTable:
    """An in-memory stand-in for the `documents` table's (source, dataset_id) -> document_id mapping.

    `get_or_create_unsafe` reproduces the pre-fix SELECT-then-INSERT race
    exactly, including a deliberate, injectable delay between the read and
    the write so the race is reliably reproducible on a single machine
    without needing real network latency to open the window.

    `get_or_create_atomic` reproduces the actual fix: one atomic
    check-and-set (Python's `dict.setdefault` under a single `threading.Lock`
    stands in for Postgres's `INSERT ... ON CONFLICT DO NOTHING RETURNING`)
    This is correct with zero external coordination, because the "check" and
    the "set" are literally the same indivisible operation.
    """

    _rows: dict[tuple[str, str], str] = field(default_factory=dict)
    _insert_lock: threading.Lock = field(default_factory=threading.Lock)
    race_window_seconds: float = 0.05

    def get_or_create_unsafe(self, source: str, dataset_id: str) -> str:
        """The pre-fix shape: SELECT, sleep (simulating real query/network latency), then INSERT."""
        key = (source, dataset_id)
        existing = self._rows.get(key)
        if existing is not None:
            return existing
        time.sleep(
            self.race_window_seconds
        )  # the race window: another thread's SELECT can land here
        new_id = str(uuid.uuid4())
        self._rows[key] = (
            new_id  # last writer wins: a lost update, or two callers get different ids
        )
        return new_id

    def get_or_create_atomic(self, source: str, dataset_id: str) -> str:
        """The actual fix's shape: one atomic check-and-set, no window for a second caller to land in."""
        key = (source, dataset_id)
        with self._insert_lock:
            if key not in self._rows:
                self._rows[key] = str(uuid.uuid4())
            return self._rows[key]


class RedisDistributedLock:
    """A minimal `SET key value NX PX ttl_ms` distributed lock (Redlock-style single-instance version).

    Deliberately does not hand-build a lock primitive Redis doesn't
    already give atomically: `SET ... NX` is itself the atomic
    "only I get this" primitive (a second `SET NX` while the key exists
    simply fails), and releasing checks the lock's own random `token`
    before deleting so this holder can never release a lock it no longer
    owns (e.g. one whose TTL already expired and was re-acquired by
    someone else). A `GET`-then-`DEL` pair would itself be racy, so
    release uses a small Lua script to make the compare-and-delete atomic
    too.

    Parameters
    ----------
    redis_client : redis.Redis
    name : str
        The lock's logical name (e.g. `"doc_ingest:security/policy.md:techfusion"`).
    ttl_ms : int
        Auto-expiry so a crashed holder can never wedge the lock forever. This follows the same reasoning `security.rate_limit`'s Redis
        keys use TTLs for, applied to mutual exclusion instead of counting.
    """

    _RELEASE_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end"
    )

    def __init__(self, redis_client: redis.Redis, name: str, ttl_ms: int = 5000) -> None:
        self._client = redis_client
        self.key = f"lock:{name}"
        self.ttl_ms = ttl_ms
        self._token: str | None = None
        self._release_script = redis_client.register_script(self._RELEASE_SCRIPT)

    def acquire(self, timeout_seconds: float = 2.0, retry_interval_seconds: float = 0.02) -> bool:
        """Block (up to `timeout_seconds`) until the lock is acquired, or return `False`.

        Returns
        -------
        bool
            `True` if acquired (holder must call `release()` when done);
            `False` on timeout.
        """
        token = str(uuid.uuid4())
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._client.set(self.key, token, nx=True, px=self.ttl_ms):
                self._token = token
                return True
            time.sleep(retry_interval_seconds)
        return False

    def release(self) -> bool:
        """Release the lock, only if this instance still holds it (token match).

        Returns
        -------
        bool
            `True` if this call actually deleted the key.
        """
        if self._token is None:
            return False
        released = bool(self._release_script(keys=[self.key], args=[self._token]))
        self._token = None
        return released

    def __enter__(self) -> RedisDistributedLock:
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock {self.key!r}")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def run_concurrent_get_or_create(
    table: RacyDocumentIdTable,
    source: str,
    dataset_id: str,
    worker_count: int,
    use_lock: RedisDistributedLock | None,
) -> list[str]:
    """Run `worker_count` concurrent `get_or_create` calls for the same `(source, dataset_id)`.

    Parameters
    ----------
    table : RacyDocumentIdTable
    source, dataset_id : str
    worker_count : int
    use_lock : RedisDistributedLock | None
        `None` uses `table.get_or_create_unsafe` directly (the race).
        A `RedisDistributedLock` instance wraps `get_or_create_unsafe` in
        `with use_lock:` (serializing it, without changing the racy
        method itself) to show the lock alone is sufficient to fix this
        specific race.

    Returns
    -------
    list[str]
        The `document_id` each of the `worker_count` calls observed. A
        correct outcome is every entry identical; the race produces more
        than one distinct value.
    """
    results: list[str | None] = [None] * worker_count

    def _call(index: int) -> None:
        if use_lock is None:
            results[index] = table.get_or_create_unsafe(source, dataset_id)
        else:
            lock = RedisDistributedLock(
                use_lock._client, use_lock.key.removeprefix("lock:"), use_lock.ttl_ms
            )
            with lock:
                results[index] = table.get_or_create_unsafe(source, dataset_id)

    threads = [threading.Thread(target=_call, args=(i,)) for i in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]
