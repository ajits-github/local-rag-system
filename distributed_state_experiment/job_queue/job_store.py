"""Persisted job status, queryable independent of the queue/stream mechanics.

A Redis hash per job (`<prefix>:job:<job_id>`) plus a small string index
(`<prefix>:idempotency:<idempotency_key>` -> `job_id`) for O(1)
submission-time dedup. This is deliberately the *simplest* thing that
satisfies "job state persisted, client can query job status" -- a real
production system would likely persist this in Postgres (this codebase
already has one, and `FeedbackStore`'s own small dedicated-pool pattern
in `rag/feedback/store.py` would be the natural template), but Redis
alone keeps this experiment's infrastructure surface to one moving part,
which is the right tradeoff for a learning exercise that already needs
Redis for the stream itself. See `README.md`'s "Why job state lives in
Redis, not Postgres" section for the fuller reasoning and what a real
system would likely do differently.
"""

from __future__ import annotations

import redis

from .models import JobRecord


class JobStore:
    """Reads and writes `JobRecord`s in Redis, plus the idempotency index.

    Parameters
    ----------
    redis_client : redis.Redis
    key_prefix : str
        Prefix for every key this store creates.
    """

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "ingest_jobs") -> None:
        self._client = redis_client
        self._prefix = key_prefix

    def _job_key(self, job_id: str) -> str:
        return f"{self._prefix}:job:{job_id}"

    def _idempotency_key(self, idempotency_key: str) -> str:
        return f"{self._prefix}:idempotency:{idempotency_key}"

    def find_by_idempotency_key(self, idempotency_key: str) -> JobRecord | None:
        """Return the existing job for `idempotency_key`, if a submission already exists."""
        job_id = self._client.get(self._idempotency_key(idempotency_key))
        if job_id is None:
            return None
        return self.get(job_id.decode() if isinstance(job_id, bytes) else job_id)

    def save(self, record: JobRecord) -> None:
        """Persist `record`, and register its idempotency-key index entry.

        The index write uses `SET ... NX` semantics implicitly via a
        plain `SET` here (last writer wins on the pointer, but the pointer
        always resolves to *a* real job for that key, which is all
        submission-time dedup needs) -- true atomicity for "only the
        first submitter wins" is enforced by `IngestionJobQueue.submit`'s
        own `SETNX`-based check, not by this method.
        """
        self._client.hset(self._job_key(record.job_id), mapping=record.to_redis_hash())
        self._client.set(self._idempotency_key(record.idempotency_key), record.job_id)

    def get(self, job_id: str) -> JobRecord | None:
        """Return the `JobRecord` for `job_id`, or `None` if unknown."""
        data = self._client.hgetall(self._job_key(job_id))
        if not data:
            return None
        return JobRecord.from_redis_hash(data)
