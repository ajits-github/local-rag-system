# Feedback loop

--8<-- "README.md:docs-feedback"

Feedback is never treated as ground truth: promoting a specific example
into a gold eval file remains a separate, deliberate, hand-authored step,
not something this endpoint or its export script does automatically.

Full design writeup: [Feedback loop](../architecture.md#feedback-loop-feedback-milestone).

API reference: [Feedback](../reference/feedback.md).

## User feedback loop (feedback milestone)

Adds a closed feedback path on top of the existing query/agent routes:
answer -> thumbs up/down (+ optional structured reason/comment) ->
`POST /feedback` -> a tenant-scoped Postgres row -> `scripts/
export_feedback.py` (JSONL/CSV) -> human review -> an optional, always
manual, hand-authored promotion into a gold eval file. Deliberately does
not implement automatic feedback-to-gold promotion or a CI eval gate;
see `docs/architecture.md`'s "Feedback loop" section for the full design
(dedup semantics, privacy posture, export/curation boundary), this is
the condensed version.

- **`request_id` is the answer/run identifier, reused rather than
  invented.** `QueryResponse`/`AgentQueryResponse` gained one additive
  field, `request_id`, sourced from the same per-request contextvar
  (`rag.logging_config.get_request_id`) that already produces the
  `x-request-id` response header. `POST /agent/query/stream`'s SSE
  terminal frame captures this value once, synchronously, in
  `agent_query_stream()` itself and threads it through as a plain
  argument, reading the contextvar deep inside the async generator
  would race against `RequestIDMiddleware`'s own `finally`-block reset,
  since that generator can run concurrently with, and outlive, what
  `call_next()`'s await covers. `request_id` is never existence-checked
  against a request log (none exists); it's an opaque, format-bounded
  correlation string a caller asserts, not a verified reference, a
  documented, deliberate choice.
- **`POST /feedback` (`api/routers/feedback.py`) reuses `/query`'s exact
  JWT-precedence rule**, via a new, smaller sibling of
  `build_authorization_context`: `rag.api.request_auth.
  resolve_trusted_tenant_id` (same forged-claim-logging behavior,
  returning a bare `tenant_id` since feedback has no ACL to build a full
  `AuthorizationContext` for). `get_current_identity` is reused
  unmodified, so `security.auth.enabled`/`insecure_dev_mode` behave
  identically to every other route. `FeedbackRequest` sets
  `extra="forbid"` (a deviation from `QueryRequest`'s permissive
  default, deliberate for this narrow write path) and a
  `model_validator` rejects a reason from the wrong rating's fixed
  vocabulary (`rag.feedback.schemas.NEGATIVE_FEEDBACK_REASONS`/
  `POSITIVE_FEEDBACK_REASONS`) as a 422.
- **`rag/feedback/store.py`'s `FeedbackStore`** is a small, dedicated
  Postgres store with its own connection pool, not folded into
  `PgVectorStore`, since feedback has no embedding/ACL concept at all.
  Dedup/update key: `(tenant_id, caller_key, request_id)`, via one
  `INSERT ... ON CONFLICT DO UPDATE ... RETURNING feedback_id,
  (xmax = 0) AS inserted` statement (atomic under concurrency).
  `tenant_id` is stored `TEXT NOT NULL DEFAULT ''`, never nullable.
  Postgres treats every `NULL` as distinct from every other `NULL` in a
  `UNIQUE` constraint, so a nullable key would never actually deduplicate
  two anonymous/no-tenant submissions (a real gotcha, caught and fixed
  before it ever reached tests; see `docs/architecture.md`). `caller_key`
  is the verified JWT subject's pseudonymous hash, or the fixed literal
  `"anonymous"` with no verified identity, a documented limitation
  (two different anonymous callers who happen to submit against the same
  `request_id`, a per-response UUIDv4 shown only to the original caller,
  share one row), not an oversight.
- **Schema lives in `scripts/init_db.py`'s `build_feedback_schema_sql`**,
  run by the same idempotent `python scripts/init_db.py`/`make up` every
  other table already goes through. `cited_source_ids`/`tool_calls` are
  `TEXT[]` columns (matching `chunks`' own array-column precedent);
  generation/prompt/retrieval/reranker lineage are normalized columns,
  read from `AppConfig` at submission time (a best-effort snapshot, not
  a per-answer record, no per-answer metadata store exists yet).
- **Privacy**: no JWT/token/`AuthorizationContext` is ever stored (only
  a plain-value `FeedbackWriteInput` dataclass reaches the store,
  test-proven); `tool_calls` is only ever public tool *names*, never
  reasoning or arguments; `query`/`answer` are the caller's own
  already-received, already-redacted text (independently toggleable off
  via `FeedbackConfig.store_query_text`/`store_answer_text`);
  `comment`/`cited_source_ids`/`tool_calls` are all runtime-config-bounded
  (`feedback.max_comment_length`/`max_cited_sources`, router-level
  checks, not Pydantic field constraints, matching `DoSLimitsConfig`'s
  established reasoning); the `feedback_submitted` audit event logs only
  IDs/enums/a boolean, never comment/query/answer content.
- **No public `GET /feedback`.** `scripts/export_feedback.py`
  (`FeedbackStore.export`, filterable by tenant/rating/dataset/route/date)
  is the reviewer-facing path, matching the established `scripts/`-based
  internal-tooling convention (`record_experiment.py`,
  `compare_experiments.py`) rather than a new admin API surface.
- **`config.feedback`** (`FeedbackConfig`): `enabled` (default `true`,
  404s the route when `false`), `table_name`, `store_query_text`/
  `store_answer_text` (both default `true`), `max_comment_length`
  (1000), `max_query_text_length` (2000), `max_answer_text_length`
  (8000), `max_cited_sources` (20).
- **Frontend**: `frontend/src/components/chat/FeedbackControls.tsx`
  renders thumbs up/down plus a disclosed-on-demand reason/comment form,
  keyed off `message.debug.requestId`; clicking an already-submitted,
  already-selected rating again is a no-op, clicking the other rating
  (an explicit change) submits an update, mirroring the backend's own
  upsert semantics client-side.
- **Observability**: `rag.observability.metrics`'s
  `FEEDBACK_SUBMISSIONS_TOTAL` (labeled `rating`/`outcome`,
  `created`/`updated`/`failed`, all fixed vocabularies) and
  `FEEDBACK_SUBMISSION_LATENCY_SECONDS`, following the exact
  dedicated-registry/`_defensive`-wrapper pattern every other metric
  already uses; a storage failure also increments the generic
  `rag_errors_total{component="feedback"}`.
