"""Job payload/status/record shapes for the distributed ingestion queue.

Kept as plain dataclasses (not Pydantic) so this experiment package has
no hard dependency on the rest of `rag.*` beyond what the worker itself
needs at processing time -- the queue/job-store machinery is reusable
even for a caller that never touches `rag.ingestion`.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    """The fixed lifecycle a job moves through.

    `SUBMITTED` -> `PROCESSING` -> (`COMPLETED` | back to `SUBMITTED` for a
    retry | `DEAD_LETTER` once retries are exhausted). A job never moves
    backwards out of `COMPLETED`/`DEAD_LETTER` -- both are terminal.
    """

    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True)
class IngestionJobPayload:
    """What a caller submits: enough to run `IngestionPipeline.ingest_file`.

    Attributes
    ----------
    source_path : str
        Path to the file to ingest (matches `IngestionPipeline.ingest_file`'s
        own `path` argument).
    dataset_id : str
        Required namespace, same non-defaulted concept as everywhere else
        in this codebase (see the root `CLAUDE.md`'s "Document identity"
        section) -- there is no "no dataset" job.
    idempotency_key : str
        `f"{dataset_id}:{source_path}:{content_sha256}"` -- reuses this
        repository's own existing `(source, dataset_id)` document-identity
        scoping plus the file's checksum, rather than inventing a second,
        parallel idempotency concept. Two submissions of the same file
        (same bytes) into the same dataset always compute the same key,
        and `IngestionPipeline`'s own checksum-gated
        `replace_document_chunks` is what makes processing that key twice
        genuinely safe (a no-op on the second run), not just this queue's
        own submission-time dedup.
    poison : bool
        Test-only flag: when `True`, the worker's simulated processing
        step always raises, regardless of whether the file exists --
        used by experiments 8-10 (deliberately failing job, retries,
        dead-letter) without needing a real corrupt file on disk.
    slow_seconds : float
        Test-only: `simulated_ingestion_process_fn` sleeps this long
        (beyond its own small base latency) before completing -- used by
        experiment 4-5's kill-a-worker-mid-processing demo to open a
        reliable window in which to kill the process while it still holds
        the job claimed but unacked, without guessing at a fixed sleep.
    submitted_at : float
        `time.time()` at submission, for latency measurement.
    """

    source_path: str
    dataset_id: str
    idempotency_key: str
    poison: bool = False
    slow_seconds: float = 0.0
    submitted_at: float = field(default_factory=time.time)

    @staticmethod
    def build(
        source_path: str,
        dataset_id: str,
        content: bytes,
        poison: bool = False,
        slow_seconds: float = 0.0,
    ) -> IngestionJobPayload:
        """Construct a payload, deriving `idempotency_key` from content bytes.

        Parameters
        ----------
        source_path : str
            Path to the file being submitted.
        dataset_id : str
            Ingestion namespace.
        content : bytes
            The file's own bytes, hashed exactly the same way
            `IngestionPipeline`/`PgVectorStore` already hash a document
            (`hashlib.sha256`) to detect unchanged content.
        poison : bool, optional
            See the class docstring.

        Returns
        -------
        IngestionJobPayload
        """
        checksum = hashlib.sha256(content).hexdigest()
        return IngestionJobPayload(
            source_path=source_path,
            dataset_id=dataset_id,
            idempotency_key=f"{dataset_id}:{source_path}:{checksum}",
            poison=poison,
            slow_seconds=slow_seconds,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string suitable for a Redis Stream field value."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str) -> IngestionJobPayload:
        """Deserialize from `to_json()`'s output."""
        return IngestionJobPayload(**json.loads(raw))


@dataclass
class JobRecord:
    """Persisted job state, queryable by a client independent of queue mechanics.

    Attributes
    ----------
    job_id : str
        Opaque id returned to the submitter (`f"job-{uuid4()}"`), distinct
        from the Redis Stream entry id (which changes on every
        re-delivery/retry `XADD`) so a client has one stable handle for
        the whole job's lifetime across retries.
    status : JobStatus
    attempts : int
        Number of processing attempts made so far (starts at 0).
    max_attempts : int
    last_error : str | None
        The exception's class name and message from the most recent
        failed attempt, if any -- never a full traceback (bounded,
        non-sensitive, matches this codebase's own `tool_call_completed`
        `error_type` convention of logging shape, not raw content).
    result : dict | None
        A small summary once `COMPLETED` (e.g. chunks embedded/reused).
    created_at, updated_at : float
    """

    job_id: str
    idempotency_key: str
    status: JobStatus
    payload_json: str = ""
    attempts: int = 0
    max_attempts: int = 5
    last_error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def new(idempotency_key: str, max_attempts: int, payload_json: str) -> JobRecord:
        """Create a fresh `SUBMITTED` record with a new `job_id`.

        `payload_json` (the original `IngestionJobPayload.to_json()`) is
        stored on the record itself so a retry can always re-`XADD` the
        exact original message without needing the caller to resubmit it.
        """
        return JobRecord(
            job_id=f"job-{uuid.uuid4()}",
            idempotency_key=idempotency_key,
            status=JobStatus.SUBMITTED,
            payload_json=payload_json,
            max_attempts=max_attempts,
        )

    def to_redis_hash(self) -> dict[str, str]:
        """Flatten to a `str -> str` mapping for `HSET`."""
        data = asdict(self)
        data["status"] = self.status.value
        data["result"] = json.dumps(self.result) if self.result is not None else ""
        data["last_error"] = self.last_error or ""
        return {k: str(v) for k, v in data.items()}

    @staticmethod
    def from_redis_hash(data: dict[bytes, bytes] | dict[str, str]) -> JobRecord:
        """Rebuild a `JobRecord` from `HGETALL`'s output (bytes or str keys/values)."""
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }
        return JobRecord(
            job_id=decoded["job_id"],
            idempotency_key=decoded["idempotency_key"],
            status=JobStatus(decoded["status"]),
            payload_json=decoded.get("payload_json", ""),
            attempts=int(decoded["attempts"]),
            max_attempts=int(decoded["max_attempts"]),
            last_error=decoded["last_error"] or None,
            result=json.loads(decoded["result"]) if decoded["result"] else None,
            created_at=float(decoded["created_at"]),
            updated_at=float(decoded["updated_at"]),
        )
