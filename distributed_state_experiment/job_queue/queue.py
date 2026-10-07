"""Redis Streams-backed ingestion job queue: producer side plus claim/ack/retry mechanics.

Uses one Redis Stream as the work queue, a consumer group so multiple
worker replicas can each claim disjoint entries, a sorted set for
delayed retries (`ZADD`/`ZRANGEBYSCORE`), and a second stream as the
dead-letter destination. See `README.md`'s "Job queue design" section
for the full write-up of each primitive and why it was chosen over the
alternatives (`XAUTOCLAIM` vs. explicit `XPENDING`+`XCLAIM`, a delayed
ZSET vs. re-`XADD`-ing future-dated stream entries directly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis

from .job_store import JobStore
from .models import IngestionJobPayload, JobRecord, JobStatus

_NEW_MESSAGES = ">"


class QueueFullError(RuntimeError):
    """Raised by `submit()` when the stream's backlog is at or above the configured cap.

    This is the queue's backpressure mechanism: a producer that keeps
    submitting into an already-saturated queue gets a clear, immediate
    error rather than silently growing an unbounded backlog that no
    worker fleet can ever catch up on.
    """


@dataclass(frozen=True)
class ClaimedJob:
    """One claimed stream entry, ready for a worker to process.

    Attributes
    ----------
    entry_id : str
        The Redis Stream entry id (changes every re-delivery; never a
        stable job handle. Use `record.job_id` for that).
    record : JobRecord
        The job's persisted state, already marked `PROCESSING` with
        `attempts` incremented for this claim.
    payload : IngestionJobPayload
        The original submission payload.
    redelivered : bool
        `True` when this claim came from `reclaim_stale()` (a previous
        consumer held it and either crashed or is still slow) rather than
        a fresh `>` read. Surfaced so a worker/test can distinguish "first
        delivery" from "at-least-once redelivery" when demonstrating
        duplicate-delivery handling.
    """

    entry_id: str
    record: JobRecord
    payload: IngestionJobPayload
    redelivered: bool = False


class IngestionJobQueue:
    """Producer + claim/ack/retry/dead-letter operations over one Redis Stream.

    Parameters
    ----------
    redis_client : redis.Redis
    stream_name : str
        The main work stream.
    group_name : str
        The shared consumer group every worker replica joins.
    key_prefix : str | None
        Prefix for the job-store, retry-zset, dead-letter-stream, and
        backlog-counter keys this queue creates alongside `stream_name`.
        Defaults to `stream_name` itself (with a trailing literal
        `":stream"` suffix stripped, if present) rather than one fixed
        global default. Two `IngestionJobQueue` instances pointed at
        *different* streams in the same Redis database must never share
        one job store/backlog counter/dead-letter stream just because
        neither caller passed `key_prefix` explicitly. This was a real
        bug in an earlier version of this class, caught by
        `distributed_state_experiment/cli/run_experiments.py`'s own live
        run: every experiment's queue silently shared one global
        `"ingest_jobs"` prefix, so `backlog_depth()` and
        `dead_letter_depth()` read cross-contaminated counts across
        unrelated experiment sections (see `README.md`'s "What was
        executed vs. simulated" section for the exact numbers this
        produced before the fix).
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        stream_name: str = "ingest_jobs:stream",
        group_name: str = "ingest_workers",
        key_prefix: str | None = None,
    ) -> None:
        if key_prefix is None:
            key_prefix = stream_name.removesuffix(":stream")
        self._client = redis_client
        self.stream_name = stream_name
        self.group_name = group_name
        self._key_prefix = key_prefix
        self.dead_letter_stream = f"{key_prefix}:dead_letter"
        self.retry_zset = f"{key_prefix}:retry_due"
        self._backlog_key = f"{key_prefix}:backlog"
        self.store = JobStore(redis_client, key_prefix=key_prefix)
        self._ensure_group(stream_name)

    def _ensure_group(self, stream: str) -> None:
        """Create the consumer group (and the stream, via `mkstream`) if it doesn't exist yet.

        Idempotent: `BUSYGROUP` (the error Redis raises for an
        already-existing group) is swallowed, matching this codebase's
        own "idempotent setup" convention (`scripts/init_db.py`'s schema
        creation is the same shape for Postgres).
        """
        try:
            self._client.xgroup_create(
                name=stream, groupname=self.group_name, id="0", mkstream=True
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def submit(
        self,
        payload: IngestionJobPayload,
        max_attempts: int = 5,
        max_queue_depth: int | None = None,
    ) -> JobRecord:
        """Submit a job, deduplicating on `payload.idempotency_key`.

        Parameters
        ----------
        payload : IngestionJobPayload
        max_attempts : int, optional
            Retry budget for this job (default 5).
        max_queue_depth : int | None, optional
            Backpressure cap: when the number of jobs still outstanding
            (`backlog_depth()`, not the stream's raw `XLEN`. See
            `queue_depth()`'s docstring for why those differ) is at or
            above this, raise `QueueFullError` instead of enqueuing.
            `None` (the default) means unbounded.

        Returns
        -------
        JobRecord
            Either a freshly created record, or the existing record for
            this exact `idempotency_key` if one was already submitted
            (submission-time dedup: experiment 6/7's "deliver the same
            job twice" is about *processing*-time duplicate delivery,
            which this dedup does not need to prevent on its own; the
            underlying `IngestionPipeline` checksum idempotency is the
            real backstop either way).

        Raises
        ------
        QueueFullError
            If `max_queue_depth` is set and the stream is already at or
            over that depth.
        """
        existing = self.store.find_by_idempotency_key(payload.idempotency_key)
        if existing is not None:
            return existing

        if max_queue_depth is not None:
            depth = self.backlog_depth()
            if depth >= max_queue_depth:
                raise QueueFullError(
                    f"Queue depth {depth} >= max_queue_depth {max_queue_depth}; "
                    "backpressure engaged, submission rejected"
                )

        record = JobRecord.new(payload.idempotency_key, max_attempts, payload.to_json())
        self.store.save(record)
        self._client.xadd(self.stream_name, {"job_id": record.job_id})
        self._client.incr(self._backlog_key)
        return record

    def claim(self, consumer_name: str, count: int = 1, block_ms: int = 2000) -> list[ClaimedJob]:
        """Claim up to `count` new entries for `consumer_name` (bounded concurrency lives in the caller).

        A plain `XREADGROUP ... STREAMS <stream> >` read: entries this
        consumer group has never delivered before. Each claimed entry is
        marked `PROCESSING` in the job store with `attempts` incremented.

        Parameters
        ----------
        consumer_name : str
            This worker replica's unique name within the group (visible
            in `XPENDING`'s per-consumer breakdown).
        count : int, optional
            Maximum entries to claim in this call.
        block_ms : int, optional
            How long to block waiting for new entries if none are
            immediately available.

        Returns
        -------
        list[ClaimedJob]
        """
        response = self._client.xreadgroup(
            groupname=self.group_name,
            consumername=consumer_name,
            streams={self.stream_name: _NEW_MESSAGES},
            count=count,
            block=block_ms,
        )
        return self._to_claimed_jobs(response, redelivered=False)

    def reclaim_stale(
        self, consumer_name: str, min_idle_ms: int, count: int = 10
    ) -> list[ClaimedJob]:
        """Reclaim entries that some consumer holds but hasn't acked in `min_idle_ms`.

        This is the worker-crash-recovery path (experiments 4-5): if a
        worker claims a job and dies before `ack`/`fail`, the entry stays
        in that consumer's Pending Entries List (PEL) forever until
        another worker explicitly reclaims it. Implemented as the
        explicit two-call `XPENDING` (list idle entries) + `XCLAIM`
        (reassign them to this consumer) pattern on purpose, even though
        Redis >= 6.2's single-call `XAUTOCLAIM` does the same thing more
        conveniently. This experiment's README explains both and why
        the explicit form is more teachable.

        Parameters
        ----------
        consumer_name : str
            The reclaiming worker's name; claimed entries become owned by
            this consumer.
        min_idle_ms : int
            Only reclaim entries that have been pending (unacked) for at
            least this long: the "visibility timeout."
        count : int, optional
            Maximum entries to inspect/reclaim per call.

        Returns
        -------
        list[ClaimedJob]
            Each with `redelivered=True`.
        """
        pending = self._client.xpending_range(
            self.stream_name, self.group_name, min="-", max="+", count=count, idle=min_idle_ms
        )
        if not pending:
            return []
        entry_ids = [
            entry["message_id"].decode()
            if isinstance(entry["message_id"], bytes)
            else entry["message_id"]
            for entry in pending
        ]
        claimed_raw = self._client.xclaim(
            self.stream_name,
            self.group_name,
            consumer_name,
            min_idle_time=min_idle_ms,
            message_ids=entry_ids,
        )
        response = [(self.stream_name, claimed_raw)]
        return self._to_claimed_jobs(response, redelivered=True)

    def _to_claimed_jobs(self, xread_response: list, redelivered: bool) -> list[ClaimedJob]:
        """Resolve raw `XREADGROUP`/`XCLAIM` entries into `ClaimedJob`s, marking each `PROCESSING`."""
        claimed: list[ClaimedJob] = []
        for _stream, entries in xread_response:
            for entry_id_raw, fields in entries:
                entry_id = (
                    entry_id_raw.decode() if isinstance(entry_id_raw, bytes) else entry_id_raw
                )
                job_id_raw = fields.get(b"job_id") or fields.get("job_id")
                job_id = job_id_raw.decode() if isinstance(job_id_raw, bytes) else job_id_raw
                record = self.store.get(job_id)
                if record is None:
                    # The job store entry is gone (should not happen in normal
                    # operation). Ack it away rather than looping on it forever.
                    self.ack(entry_id)
                    continue
                record.attempts += 1
                record.status = JobStatus.PROCESSING
                record.updated_at = time.time()
                self.store.save(record)
                payload = IngestionJobPayload.from_json(record.payload_json)
                claimed.append(ClaimedJob(entry_id, record, payload, redelivered=redelivered))
        return claimed

    def ack(self, entry_id: str) -> None:
        """Acknowledge `entry_id`, removing it from the consumer group's PEL."""
        self._client.xack(self.stream_name, self.group_name, entry_id)

    def complete(self, claimed: ClaimedJob, result: dict) -> None:
        """Mark a job `COMPLETED`, ack its stream entry, and decrement the live backlog counter."""
        claimed.record.status = JobStatus.COMPLETED
        claimed.record.result = result
        claimed.record.updated_at = time.time()
        self.store.save(claimed.record)
        self.ack(claimed.entry_id)
        self._client.decr(self._backlog_key)

    def fail(
        self,
        claimed: ClaimedJob,
        error: Exception,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        """Handle a processing failure: schedule an exponential-backoff retry, or dead-letter.

        Always `ack`s the current stream entry first (successful or not,
        this specific delivery attempt is over). A retry is a brand
        new stream entry added once its backoff elapses
        (`promote_due_retries`), not the same entry left pending.

        Parameters
        ----------
        claimed : ClaimedJob
        error : Exception
            The exception raised while processing. Only its class name
            and message are stored (`JobRecord.last_error`), never a full
            traceback.
        base_backoff_seconds, max_backoff_seconds : float, optional
            `min(base * 2**(attempts-1), max)`: doubles each attempt,
            capped.
        """
        record = claimed.record
        record.last_error = f"{type(error).__name__}: {error}"
        record.updated_at = time.time()

        if record.attempts >= record.max_attempts:
            self._move_to_dead_letter(record)
            self.ack(claimed.entry_id)
            return

        backoff = min(base_backoff_seconds * (2 ** (record.attempts - 1)), max_backoff_seconds)
        record.status = JobStatus.RETRY_SCHEDULED
        self.store.save(record)
        self._client.zadd(self.retry_zset, {record.job_id: time.time() + backoff})
        self.ack(claimed.entry_id)

    def _move_to_dead_letter(self, record: JobRecord) -> None:
        """Persist `record` as `DEAD_LETTER`, copy it onto the dead-letter stream, decrement backlog."""
        record.status = JobStatus.DEAD_LETTER
        self.store.save(record)
        self._client.decr(self._backlog_key)
        self._client.xadd(
            self.dead_letter_stream,
            {
                "job_id": record.job_id,
                "idempotency_key": record.idempotency_key,
                "attempts": str(record.attempts),
                "last_error": record.last_error or "",
            },
        )

    def promote_due_retries(self, now: float | None = None) -> int:
        """Move any retry whose backoff has elapsed back onto the main stream.

        Meant to be called once per worker loop iteration (cheap: one
        `ZRANGEBYSCORE` when nothing is due). Returns the number of jobs
        promoted, for the caller's own metrics/logging.

        Parameters
        ----------
        now : float | None, optional
            Override "now" for deterministic tests; defaults to
            `time.time()`.
        """
        now = time.time() if now is None else now
        due_job_ids = self._client.zrangebyscore(self.retry_zset, min="-inf", max=now)
        promoted = 0
        for raw_job_id in due_job_ids:
            job_id = raw_job_id.decode() if isinstance(raw_job_id, bytes) else raw_job_id
            removed = self._client.zrem(self.retry_zset, raw_job_id)
            if not removed:
                # Another worker's promote_due_retries already claimed this
                # job_id between our ZRANGEBYSCORE and ZREM. ZREM's return
                # value is itself the atomic "did I win" check, so skip
                # re-adding it a second time.
                continue
            record = self.store.get(job_id)
            if record is None:
                continue
            record.status = JobStatus.SUBMITTED
            self.store.save(record)
            self._client.xadd(self.stream_name, {"job_id": job_id})
            promoted += 1
        return promoted

    def queue_depth(self) -> int:
        """Return the main stream's `XLEN`: total entries ever added, not "unprocessed backlog."

        A real gotcha worth stating plainly, since it is easy to assume
        otherwise: `XACK` removes an entry from the consumer group's
        Pending Entries List, but it does **not** delete the entry from
        the stream itself, so `XLEN` never shrinks just because work was
        completed. It only shrinks via explicit trimming (`XTRIM`/
        `MAXLEN`) or `XDEL`. Use `backlog_depth()` for "how much
        unfinished work exists right now," which is what
        `job_queue.metrics.QUEUE_DEPTH` actually reports.
        """
        return self._client.xlen(self.stream_name)

    def backlog_depth(self) -> int:
        """Return the number of jobs submitted but not yet in a terminal state.

        A plain Redis counter incremented on `submit()` and decremented on
        `complete()`/dead-lettering, deliberately not derived from
        `XLEN` (see that method's docstring for why `XLEN` alone cannot
        answer this). This is what a queue-depth dashboard/alert should
        actually watch.
        """
        value = self._client.get(self._backlog_key)
        return int(value) if value is not None else 0

    def dead_letter_depth(self) -> int:
        """Return the dead-letter stream's current length."""
        return self._client.xlen(self.dead_letter_stream)

    def pending_count(self) -> int:
        """Return the consumer group's total pending (claimed-but-unacked) entry count."""
        summary = self._client.xpending(self.stream_name, self.group_name)
        if not summary:
            return 0
        return int(summary["pending"])

    def heartbeat(self, consumer_name: str, ttl_seconds: int = 15) -> None:
        """Register `consumer_name` as alive for `ttl_seconds` (for `active_worker_count()`).

        A worker calling this periodically (e.g. once per main-loop
        iteration) is what makes "how many workers are actually up right
        now" observable across processes/replicas without a separate
        service registry. The TTL means a crashed worker's presence
        simply expires rather than needing an explicit deregistration
        step (the same reasoning this queue's Redis keys already use TTLs
        for elsewhere, e.g. `RedisDistributedLock`).
        """
        self._client.setex(
            f"{self._key_prefix}:worker_heartbeat:{consumer_name}", ttl_seconds, "alive"
        )

    def active_worker_count(self) -> int:
        """Return the number of workers with a live (unexpired) heartbeat."""
        return len(list(self._client.scan_iter(match=f"{self._key_prefix}:worker_heartbeat:*")))
