# Detailed Architecture

This is the expanded system view: current design, invariants, and known
limitations per subsystem. The README keeps a smaller public-facing view.
Implementation history, experiment results, and bug write-ups live in
`PROJECT_JOURNAL.md`, `KNOWN_ISSUES.md`, and `experiments/reports/`.

![Local RAG System architecture: clients and entry points, ingestion and
knowledge preparation, retrieval and evidence sources, the RAG/agent
runtime router, the context engineering layer, generation and synthesis,
feedback/evaluation/safety, and observability and
operations](assets/architecture-diagram-detailed.png)

## Retrieval Cutoff Semantics

`RetrievalPipeline.retrieve()` has three independent cutoffs:

- `retrieval.candidate_k`: candidates fetched from the vector store (per
  branch, in hybrid mode) before reranking.
- `reranker.top_n`: candidates a real reranker (`cross_encoder`/`cohere`)
  keeps after rescoring. Inert for `reranker.provider: none`;
  `NoOpReranker` is a true identity.
- `retrieval.generation_context_top_n`: primary chunks selected for
  generation, applied once, after the optional rerank and before
  relationship expansion. The only generation-size truncation in
  `retrieve()`.

Relationship expansion operates on the already-truncated primary list and
appends after it, so it never affects Recall@K/MRR computed from a
broad-enough override (see `eval/run_eval.py`).

## Multimodal + Relationship-Aware Ingestion

Images are first-class searchable elements, and structural relationships
(image to caption, table/code/image to parent section, element to
neighbor) are preserved on the existing `Chunk`/`ChunkMetadata` model, not
a separate element schema or graph database.

### Element model

`StructuredMarkdownChunker` treats a standalone `![alt](path)` line (its
own line, real asset extension) like a fenced block: it flushes pending
prose and becomes its own span with `content_type="image"`. An immediately
following emphasis-wrapped paragraph (`*Figure 1: ...*`) is folded into
that span, so an image and its caption are always retrieved together; a
caption is never independently addressable. Inline image links inside
prose stay attachment tags on that prose chunk.

### `parent_chunk_id`

Nullable `chunks` columns: `parent_chunk_id`, `vision_generated`,
`vision_description`. `Writer.write` computes `parent_chunk_id` in its
single pass: a non-prose span (table/code/configuration/chart/image)
points at the most recent **prose** span with the same `section_path`.
Prose spans, and non-prose spans that open a section, get `None`
(a section's prose is reconstructed via `VectorStore.get_chunks_by_section`
instead). It's a heuristic, not a guarantee for every document shape. No
FK constraint: parent and child land in the same insert batch and are
always deleted together per document.

### Vision

`config.vision.provider` is `none` by default: no image bytes are read,
and an image's indexable content is its alt text plus folded caption. With
`ollama` (local `OllamaVisionProvider`, see
[Layout-Aware Document Ingestion and Vision](#layout-aware-document-ingestion-and-vision)),
ingestion resolves each image's `source_anchor` to a file, checks the
Postgres `image_description_cache`, calls the provider only on a miss, and
adds a separate **sibling** chunk (`vision_generated=True`) rather than
overwriting the caption/alt-text chunk.

### Relationship-aware context expansion

`retrieval.relationship_expansion` (`enabled: false` by default) adds a
post-rerank step: for each ranked result, its parent and/or immediate
previous/next chunk in the same section (`get_chunks_by_ids`,
`get_chunks_by_section`) are **appended** after the ranked list, tagged
`origin="expanded"` and `expanded_from=<chunk id>`. An expanded result
inherits its originator's score; check `origin`, not `score`. Results are
deduplicated and capped at `max_related_elements` per originator. Every
lookup is scoped to the originator's `document_id`, so expansion can't
cross a `dataset_id` boundary by construction.

Prompt source labels include section, `content_type`, whether an image
chunk is caption-only or vision-generated, and whether it was expanded,
so the model is never given grounds to imply it inspected an image it
didn't.

### Evaluation ground truth

`GoldExample` has optional `content_type`, `reference_contexts`,
`reference_visual_contexts`, `relevant_images`, `relevant_sections`,
`requires_vision`, `requires_relationship_expansion`; older gold files
still parse. `reference_contexts` are verbatim corpus excerpts;
`reference_visual_contexts` are facts visible only in an image. Both are
read **only** by `eval/*.py`; `ingestion`/`retrieval`/`generation` never
import `rag.eval`, enforced by `test_gold_data_isolation.py`, plus a
runtime test that a `reference_visual_contexts` marker never reaches a
prompt.

`reference_context_is_supported` is a normalized substring check against
raw source text. It proves the text exists in the source, not that it's
the correct supporting passage, and formatting drift can under-count.

### Deterministic metrics (`eval/run_eval.py`)

All from the same top-10 retrieval used for Recall@10, no LLM involved:

- `content_type_breakdown`: Recall/hit-rate by the gold file's authored
  `content_type`.
- `reference_context_analysis`: A = document and supporting context
  retrieved, B = document retrieved but context missed, C = document
  missed; `supporting_context_hit_rate = A / (A + B)`.
- `relevant_image_hit_rate`: whether the right image element was
  retrieved.
- `relationship_expansion_contribution_rate`: among
  `requires_relationship_expansion` rows, how often an expanded chunk
  supplied context the primary set lacked. Always 0 with expansion off.
- `vision_behavior_breakdown` (generation runs): heuristic triage of
  `requires_vision` rows into `correct_refusal` / `hallucinated_answer` /
  `caption_leak_success` / `incorrect_or_missing`. A phrase-match
  heuristic with known false negatives.

### RAGAS judge-call caching

RAGAS's own `DiskCacheBackend` key never includes the judge model, so a
shared cache could replay one model's verdict for another.
`eval/ragas_cache.py` namespaces keys by judge provider, model,
temperature, max_tokens, and `ragas` version.

## Authorization, Freshness, and Trust (safety/freshness milestone)

Retrieval-time tenant/role authorization, document version freshness,
prompt-injection detection, and trust filtering. The gate is SQL predicates
in `PgVectorStore`, applied before any row leaves Postgres. The prompt rules
in `rag_answer_v3.yaml` are defense-in-depth, not the gate.

### Identity model at this layer

This layer enforces authorization **given** an identity context; it does
not authenticate. Authentication (JWT verification) is in `api/auth.py`,
see [Authenticated API Boundary and Security Hardening](#authenticated-api-boundary-and-security-hardening-auth-boundary-milestone).
With a verified identity, `tenant_id`/`roles` come from JWT claims and any
request-body values are ignored. `as_of` and `require_trust_level` remain
caller-supplied query controls. `eval/run_eval.py` builds the context from
trusted gold-file fields, since the eval harness has no HTTP boundary.

### AuthorizationContext and the enforcement predicate

`AuthorizationContext` (`retrieval/authorization.py`: `tenant_id`, `roles`,
`as_of`, `include_superseded`, `require_trust_level`,
`resolved_excluded_document_ids`) is passed explicitly to
`VectorStore.search`/`search_keyword`/`get_chunks_by_ids`/
`get_chunks_by_section`. It is never built from the caller-controlled
`filters` dict (`ALLOWED_FILTER_FIELDS`), which can only narrow results.

`build_authorization_where_clause` (`vectorstore/pgvector.py`) is ANDed onto
every query:

```sql
(
    tenant_id IS NULL                                    -- untenanted legacy content: never gated
    OR tenant_id = %(caller_tenant)s
    OR (allowed_roles IS NOT NULL
        AND allowed_roles && %(caller_support_roles)s)   -- explicit per-document support grant
)
AND (
    allowed_roles IS NULL
    OR allowed_roles && %(caller_roles)s                 -- role membership, independent of tenant match
)
AND NOT (document_id::text = ANY(%(freshness_excluded_ids)s))   -- only when set
AND (trust_level IS NULL OR trust_level = %(required_trust_level)s)  -- only when set
```

`caller_support_roles` = `caller.roles ∩ security.authorization.cross_tenant_support_roles`
(default `["techfusion_support"]`), computed in Python first. Cross-tenant
access therefore requires a role that is the caller's own, a configured
support role, **and** listed on that document's `allowed_roles`. Tenant
match alone is never sufficient. Covered by
`tests/integration/test_authorization_isolation.py`.

Untenanted legacy chunks (`tenant_id IS NULL`) are never gated, so the
feature is additive. `security.authorization.enabled` (default `false`) is
a coarser switch: when false, `RetrievalPipeline` always passes `auth=None`.
Relationship expansion receives the same `AuthorizationContext`.

### Freshness: deterministic version-family resolution

`loaders/text_loader.py` parses governance front matter (`tenant_id`,
`allowed_roles`, `classification`, `status`, `document_version`,
`effective_from`, `trust_level`, `doc_source_type`, `supersedes`) onto every
chunk. (`doc_source_type` is a trust-provenance label, distinct from the
loader's file-type `source_type`.)

`retrieval/freshness.py` links documents into families via `supersedes`
(a filename, matched by path suffix at query time) using union-find, any
chain depth:

- **Current queries** (`as_of=None`): prefer `status="active"` members; a
  family with no active member excludes nothing.
- **Historical queries** (`as_of` set): the member with the latest
  `effective_from <= as_of`; a member with no `effective_from` is never
  excluded.

The exclusion set is computed once per query (`resolve_auth`, scoped by
`dataset_id`) into `resolved_excluded_document_ids`, so freshness and
authorization share one predicate.

### Prompt injection: detection is telemetry, not the gate

`detect_injection` (`retrieval/injection_detection.py`, a literal
phrase/regex heuristic) never blocks or drops anything. It annotates a
flagged chunk's source label ("possible embedded instruction") and feeds
the `prompt_injection_success_rate` eval metrics. Authorization is the real
defense. `rag_answer_v3` is the first prompt to make the instruction/data
boundary explicit; `config/default.yaml` still uses `v1`, and v3 is
activated by `config/experiments/secure-rag-baseline-v1*.yaml`.

### Trust: don't discard untrusted content, gate it on request

`trust_level` is stored and filterable but not a default gate, so a
poisoned document stays retrievable and "was it rejected" stays testable.
`require_trust_level` (from gold or `POST /query`) adds the NULL-permissive
`trust_level` clause only when a query asks for it.

### Ingestion: deleted-document detection and aggregate statistics

Incremental ingestion keys on `(source, dataset_id)` plus checksum: an
unchanged file is never rechunked or re-embedded. For directory targets,
`ingest_path` diffs a pre-run `list_document_sources(dataset_id)` snapshot
against discovered files and deletes the difference
(`delete_documents_by_source`); single-file targets never delete. The
snapshot is first filtered to sources under the ingestion root
(`_source_is_under_root`: resolved-path membership, separator-normalized),
so ingesting one directory never deletes documents from another root or
from `POST /ingest`. Returns `IngestionStats`.

Residual limitation: `document_id` is keyed on the literal `source` string,
so the same file found under a differently spelled root is deleted and
recreated (identity churn, no duplication or content loss). See
`KNOWN_ISSUES.md`.

### Corpus lineage and MLflow dataset tracking

`eval/corpus_lineage.py` records per run: `dataset_id`, caller-supplied
`corpus_version`, document/chunk/image/tenant counts, active/superseded
counts, gold record count and sha256, and a `corpus_digest` (sha256 over
sorted `"{source}:{checksum}"`). It goes into the eval report, the
experiment record, and MLflow (params/tags plus a best-effort
`mlflow.log_input` dataset).

### Known limitations (deliberate, not hidden)

- Freshness only links documents via an authored `supersedes`; no family
  is inferred from naming. Missing `status`/`effective_from` data degrades
  to "nothing excluded", never a guess.
- `classification` is stored but not gating; `allowed_roles` is the only
  per-document ACL, to avoid two ACLs that could disagree.
- `detect_injection` and the safety metrics are deterministic heuristics
  for triage and regression tracking, not semantic judges.
- Field-level redaction is covered in the next section.

## Field-Level Sensitive-Data Redaction (field-level-safety milestone)

Document ACL decides whether a caller may retrieve a document. Some
documents a caller may legitimately read still contain one restricted
value (e.g. an administrator credential in a tenant's own runbook). This
layer redacts such values without changing document-level authorization.

### Threat model

1. **Document authorization**: can the caller retrieve the document? (ACL,
   previous section; the primary boundary.)
2. **Field disclosure**: can the caller see a specific sensitive value
   inside an authorized document? (This section.)
3. **Encoding/transformation attacks** (spelled out, reversed, Base64,
   split): handled as a consequence of #2's enforcement point, not by a
   separate detector.

### `SensitiveFieldPolicy`

`retrieval/field_policy.py` defines a short, hardcoded
`DEFAULT_FIELD_POLICIES` list (dataset-specific, deliberately not config
and not a general secret scanner):

```python
class SensitiveFieldPolicy(BaseModel):
    field_id: str  # e.g. "synthetic_admin_credential"
    sensitivity_type: str  # e.g. "credential"
    detector: Literal["regex"] = "regex"  # swap point for a future detector kind
    pattern: str  # regex over chunk text
    allowed_roles: list[str] = Field(default_factory=list)
    redaction_marker: str = "[REDACTED:SENSITIVE_FIELD]"
```

`synthetic_admin_credential` matches both tenants' credential shapes
(`SYNTHETIC_ONLY_\w+`, `SYNTHETIC_BETA_TOKEN_\w+`); an unused
`synthetic_internal_token` policy exists as structure for future detectors.
`TF-SYNTH-*` correlation IDs are intentionally not policed (operators need
them). Since document ACL runs first, a policy's `allowed_roles` is only
evaluated within already-authorized chunks; no per-tenant policy copies.

### Ingestion-time tagging vs. query-time enforcement

- **Ingestion** (`ingestion/writer.py`): `detect_sensitive_field_ids`
  (pattern only, no role check) persists matching ids to
  `chunks.sensitive_field_ids`, so untagged chunks are never re-scanned.
- **Query** (`RetrievalPipeline._redact_sensitive_fields`):
  `redact_sensitive_fields(text, roles)` replaces matches with the marker
  unless the caller's roles intersect the policy's `allowed_roles`. Runs
  once in `_retrieve_timed` (shared by `retrieve()`, `answer()`, and the
  eval harness), after relationship expansion and before injection
  flagging. Every consumer (prompt, `sources`, eval checks, RAGAS
  `retrieved_contexts`) sees redacted text. Redaction copies the `Chunk`
  (`model_copy`) rather than mutating it, since expansion caches can share
  instances.

### Why this defeats encoding attacks

Redaction happens on retrieved `content` before the prompt is built, so an
unauthorized caller's raw value never reaches the model; there's nothing
to encode or reverse. `test_field_level_redaction.py::test_encoding_or_splitting_the_query_cannot_change_what_is_retrieved`
shows Base64, character-by-character, and reversed queries all retrieve
identical redacted text.

### Fail-closed on missing identity

Document-level `auth=None` means unrestricted (legacy corpus
compatibility). Field redaction does **not** inherit that:
`field_redaction.enabled` is independent of `authorization.enabled`, and
when on, a missing identity resolves to an empty role list, which
satisfies no policy, so every tagged field is redacted. With
`field_redaction.enabled: false` (default) behavior is unchanged. Guarded
by `test_field_redaction_fails_closed_when_no_authorization_context_supplied`.

### Composition with other controls

Field disclosure is a plain function over `roles`, not a field on
`AuthorizationContext`. Tenant ACL, document ACL, freshness, trust, and
field disclosure stay separate controls applied in sequence. Expanded
chunks are redacted by the same pass as directly retrieved ones
(`test_relationship_expanded_parent_chunk_is_also_redacted`).

### Metrics (`eval/run_eval.py` `safety` section)

- `document_unauthorized_retrieval_rate`: excludes documents that are
  both forbidden and allowed for the example (field-restricted, not
  ACL-restricted).
- `sensitive_data_leakage_rate`: literal scan of the final answer.
- `sensitive_data_authorized_disclosure_accuracy`: authorized callers who
  did receive the value (over-redaction check).
- `sensitive_data_false_redaction_rate`: redactions where the caller was
  authorized; 0 by construction, a regression guard.
- `encoded_extraction_success_rate`: value recoverable from an answer via
  direct, reversed, or Base64-decoded scan.
- Per-example `field_level_evidence` (booleans and field ids only, never
  raw values) makes non-zero rates diagnosable.

Because redaction happens at the single retrieval choke point,
experiment JSON, reports, MLflow artifacts, and the RAGAS judge payload all
inherit redacted text; no separate scrubbing step exists or is needed
(`test_answer_sources_never_contain_raw_value_for_unauthorized_caller`).

### Known limitations (deliberate, not hidden)

- Policies are a short, hardcoded, dataset-specific pattern list, not a
  general-purpose secret or PII scanner.
- `allowed_roles` is per policy, not per document. A document needing a
  different role set for the same pattern would need a new `field_id` or a
  scoping extension (not built). `detector` has only the regex
  implementation.
- `encoded_extraction_success_rate` proves specific transforms didn't
  leak, not that no transform could.

## Authenticated API Boundary and Security Hardening (auth-boundary milestone)

Verifies caller identity at the HTTP boundary (previously `tenant_id`/`roles`
were read unverified from the request body), plus adjacent hardening:
metadata leakage, untagged duplicate secrets, system/evidence separation
in generation, hosted-provider egress, and request size/rate limits.

### Identity model: authentication vs. authorization

```
HTTP request
  -> JWT verification (api/auth.py:verify_jwt)
  -> VerifiedIdentity
  -> AuthorizationContext (api/request_auth.py:build_authorization_context)
  -> retrieval ACL / freshness / trust (RetrievalPipeline, pgvector.py)
  -> field-level redaction (field_policy.py)
  -> prompt construction (separate system and user turns, see below)
```

`RetrievalPipeline` never builds an `AuthorizationContext` itself; it
receives one from the caller boundary (the API routers, or the eval
harness). Authentication therefore lives entirely in the API layer and
authorization stays unchanged underneath it.

### JWT verification

HS256 by default; RS256/ES256 selectable via `JWTConfig.algorithm` for a
future external IdP. Uses `pyjwt` with an explicit `algorithms=[...]`
allow-list (blocks `alg: none` / algorithm confusion). `verify_jwt` checks
signature, `exp` (with `leeway_seconds`), issuer and audience when
configured, and `required_claims` (default `sub`, `tenant_id`, `roles`).
Failures raise `AuthenticationError(reason=...)` with a fixed reason
literal, never the token or a raw library message.
`scripts/issue_dev_token.py` mints tokens for local testing only.

### Fail-closed, not fail-open

`security.auth.enabled: false` (default) changes nothing. When `true`:

- No `Authorization` header: 401, unless `insecure_dev_mode: true`, which
  relaxes only the no-header case.
- Invalid token (signature, expiry, malformed, missing claim, issuer,
  audience): **always** 401, regardless of `insecure_dev_mode`. A failed
  verification never proceeds with `auth=None`
  (`test_insecure_dev_mode_still_rejects_an_invalid_present_token`,
  `test_invalid_jwt_never_falls_back_to_unrestricted_retrieval`).
- With a verified identity, `AuthorizationContext` comes only from
  `identity.tenant_id`/`identity.roles`. Body values are ignored; a
  mismatch is logged as `forged_claim_attempt`.

### Audit logging

`src/rag/audit.py` (top level, since `retrieval/` must not import `api/`)
wraps the `rag.audit` logger with the existing JSON formatter and request
id. Events: `auth_success`, `auth_failure`, `authorization_denied`,
`cross_tenant_attempt`, `field_redaction_applied`,
`trust_policy_rejection`, `freshness_version_selected`,
`injection_flagged`, `forged_claim_attempt`, `rate_limit_exceeded`,
`oversized_request_rejected`, `egress_policy_blocked`. Events carry only
ids, categories, and counts: never JWT contents, secrets, chunk text, or
API keys. `pseudonymous_subject` hashes `sub` before logging.

`authorization_denied` is emitted when an active authorization context
returns fewer results than requested and
`count_excluded_by_authorization` reports excluded chunks. It carries a
count, not which documents were excluded. `cross_tenant_attempt` is
emitted at ingestion time (below); at retrieval time it's approximated by
`forged_claim_attempt`.

### Ingest-time tenant governance

Loaders without front matter (PDF/DOCX/HTML, plain Markdown) produce no
`tenant_id`, and `tenant_id IS NULL` is visible to every tenant.
`ingestion/governance.py:resolve_ingest_tenant_id` closes this for
authenticated uploads: a missing `tenant_id` is stamped with the caller's
tenant; a matching one is kept; a different one is accepted only for a
`cross_tenant_support_roles` holder, else 403 (`IngestGovernanceError`)
before any write. `ingest_file(caller=None)` (CLI, unauthenticated
requests) is unchanged.

### Atomic upload persistence

`api/routers/ingest.py:_ingest_upload_atomically` stages each upload in a
unique sibling directory of the destination (same filesystem, same
filename, so loader-derived titles and image names are correct). Size
validation, parsing, governance, and the DB write all run against the
staged file. Only after full success are extracted assets moved into
`UPLOAD_DIR/assets/` and the file installed with `os.replace`; any failure
deletes the staging directory, leaving a previously accepted file intact.
`ingest_file(source_override=...)` persists the stable destination path as
`source`, so re-uploads keep the same `document_id`
(`test_upload_staging_no_temp_leak.py`). Vision reads image bytes from the
staged file's own directory (`Writer.write(asset_base_dir=...)`), never from
the destination, so a replacement upload is never described from the
previous version's image; only the citation-facing path uses the
destination.

### Metadata and citation protection

A forbidden document never enters `results` (the SQL predicate runs
first), so its metadata can't leak (`test_metadata_leakage.py`). For
permitted documents, `field_policy.redact_source_metadata` applies the
same policies to `attachment_name` and `section_path`, inside the same
redaction pass, so every consumer gets clean metadata. The public
`SourceItem` exposes only `chunk_id`/`document_id`/`source`/`category`/`score`.

### Duplicate-secret detection

`field_policy.find_duplicate_sensitive_occurrences` groups pattern matches
by sha256 of the matched value (never the raw value) and flags groups that
span several chunks or include a chunk missing the `sensitive_field_ids`
tag (which would let query-time redaction skip it). It is a diagnostic,
not a gate: run via `scripts/detect_duplicate_sensitive_values.py` and the
always-reported `duplicate_sensitive_field_miss_rate` eval metric.

### Prompt-injection handling: structure, not more regexes

`LLM.generate(system, user)` sends system instructions and
evidence-plus-query as separate chat turns (`OllamaLLM` uses Ollama's chat
endpoint). This role separation is the real hardening. `_INJECTION_PATTERNS`
also covers some obfuscated phrasings, but remains telemetry, not a gate.
`rag_answer_v4.yaml` (explicit two-turn wording, a rule for the redaction
marker) exists but isn't active in any config, so its effect can be
measured separately from the structural change. RAGAS's adapter passes its
single prompt as `user` with an empty `system`.

### Provider-egress policy

`retrieval/egress_policy.py:apply_egress_policy(source, config)` (re-exported
from `rag.eval.egress_policy`) returns an `EgressDecision`, checking in
order:

1. `blocked_tenant_ids` deny-list.
2. `classification_policy`, which **fails closed** for an unlisted
   classification; `confidential`/`restricted` are blocked by default.
   Missing classification (legacy content) is allowed.
3. `require_authoritative_trust`: only `trust_level == "authoritative"`.
4. `block_unredacted_sensitive_fields`: a source whose
   `sensitive_field_ids` aren't all in `redacted_field_ids` is blocked.

`security.egress_policy.enabled: false` by default. Applied at the hosted
egress points:

- **Hosted reranker** (`reranker.provider: cohere`): candidates are
  field-redacted first, then filtered by egress policy, before the
  reranker call.
- **Hosted RAGAS judge** (`run_ragas_eval.py`, `run_agent_ragas_eval.py`):
  if any source of a row is blocked, the **whole row** is dropped,
  answer included, since the answer may quote the blocked content. Only a
  count is audit-logged.

Production generation is `ollama` only (`generation.provider` is a
single-value literal), so it is not an egress point. A source with no
`classification` egresses only while `allow_unclassified` is `true`
(`config/production.yaml` sets it `false`). Graph-derived results carry
their source document's current Postgres governance metadata, so they
pass through the same checks as chunks.

### Input size / DoS limits and rate limiting

`DoSLimitsConfig` (`max_query_length: 2000`, `max_top_k: 20`,
`max_filters_bytes: 4096`, `max_upload_bytes: 25 MiB`) is enforced by
`request_auth.enforce_dos_limits` (shared by `/query`, `/agent/query`,
`/agent/query/stream`), reading runtime config. `top_k < 1` is rejected
with 422. Uploads are size-checked while streaming (413, partial file
deleted). `/agent/query/stream` validates in a dependency declared before
its other dependencies, so an invalid request is rejected before a 200 is
committed and before heavy singletons load.

Rate limiting: `slowapi`, in-memory, `RateLimitConfig.enabled: false` by
default, keyed by verified `tenant_id` or else client IP. The limiter is
built once per process and bound to routes at import time, so config
changes need a restart. State is process-local: with N replicas the
effective limit is roughly N times higher (shared storage would be needed
for multi-replica deployment).

### Safety metrics (`eval/run_eval.py`)

- `unauthorized_metadata_leakage_rate`: also checks `attachment_name` and
  `section_path`.
- `provider_egress_policy_violation_rate`: sources with unredacted
  sensitive fields, independent of whether egress policy was enabled.
- `forged_role_acceptance_rate`: calls
  `api.routers.query._build_authorization_context` with a forged body and
  a verified identity; 0 by construction (regression
  guard). Requires `config` to be passed.
- `duplicate_sensitive_field_miss_rate`: corpus-level, always reported.
- `authentication_failure_acceptance_rate`,
  `oversized_request_rejection_accuracy`: opt-in HTTP probes
  (`--include-api-probes`), since they're trivial with auth disabled.

The gold-driven eval calls `RetrievalPipeline` directly, so
`security.auth.enabled` doesn't affect its core metrics.

### Known limitations (deliberate, not hidden)

- `authorization_denied` reports a count of excluded chunks, not which
  documents were excluded or why.
- Rate limiting is per process.
- Duplicate-secret detection reports; it doesn't redact.
- Injection patterns are a small literal list; role separation is the
  defense.
- No token issuance, refresh, or revocation: this layer verifies
  externally minted JWTs.

## Agentic RAG

Simple questions take the fixed `RetrievalPipeline.answer()` path;
complex or multi-hop questions get decomposition, tool selection,
evidence-sufficiency checking, and bounded retry. Every security control
(JWT identity, tenant/role authorization, freshness, trust, field
redaction, injection flagging, audit logging) stays in force; see
"Security invariants" for how.

### Graph

```
START
  │
  ▼
classify_query ── LLM JSON decision ──► query_type ∈ {simple, complex}
  │
  ├── simple ──► classic_rag (RetrievalPipeline.answer(), unchanged) ──► final
  │
  └── complex ──► decompose ──► select_tool ──► execute_tool ──► evaluate_evidence
                                    ▲                                  │
                                    │                     sufficient ──┼──► synthesize ──► final
                                    │                                  │
                                    └──────── reformulate ◄── insufficient (bounded retry)
                                                                        │
                                                     (any bound reached)┴──► synthesize_or_insufficient ──► final
```

`run_agent()` (`agent/graph.py`) is a plain bounded loop over node
functions taking and returning `AgentState`, not LangGraph (an
experiment branch explores that). Three independent bounds, each with its
own `termination_reason`: `max_agent_steps` (default 8),
`max_retrieval_attempts` (2), `max_tool_calls` (6). On any bound the run
synthesizes from gathered evidence, or returns an insufficient-evidence
response if there is none. A simple question costs exactly one extra LLM
call (classification).

### State model

`AgentState` (`agent/state.py`) holds the query, `authorization_context`,
`filters`, subquestions, evidence, tool-call history, counters, token
counts, and the final answer/citations/termination reason. It contains an
`AuthorizationContext`, never the raw `VerifiedIdentity` (which carries the
JWT `sub`). `filters` (including `dataset_id`) is set once from the request
and never changed by an LLM decision.

### Tools

![Agent tool architecture: the in-process agent dispatches local RAG and
optional graph tools directly in Python, and business tools through its
MCP client to the internal /mcp server, which resolves identity, verifies
the internal JWT, and dispatches to the case backend](assets/agent-tool-architecture.png)

The agent has three tool families. Local RAG and graph tools are direct
Python calls; business tools go through the MCP client (see
[MCP Integration](#mcp-integration)). None reimplements retrieval or
security logic (`agent/tools.py`):

| Tool | Calls | Bounding |
|---|---|---|
| `search_knowledge_base` | `RetrievalPipeline.retrieve()` | `top_k` is `Field(ge=1, le=20)`, then clamped to `agent.max_tool_top_k` |
| `get_document` | `VectorStore.get_chunks_by_source` | SQL ceiling (`max_chunks_per_document_fetch_hard_ceiling`, 50), then relevance selection to `max_chunks_per_document_fetch` (10); not LLM-writable |
| `get_latest_document` | `list_document_versions` + `freshness.resolve_current_document_source` + `get_chunks_by_source` | same as `get_document` |
| `get_related_context` | `RetrievalPipeline.expand_with_relationships()` | `relationship_expansion.max_related_elements` |
| `search_knowledge_graph`, `get_entity`, `get_neighbors`, `find_dependency_path` | `GraphRetriever` / `Neo4jClient` | bounded string lengths; `get_neighbors.max_hops` is `Field(ge=1, le=3)` and clamped again by `GraphRetriever` |
| `get_customer_case`, `get_case_status`, `update_case_status` | MCP client to `rag.mcp.business.store` | see [MCP Integration](#mcp-integration) |

Graph tools are offered (via `agent.tool_select_graph_prompt_path`) only
when `graph.enabled` is true; `GraphRetriever` applies the caller's
authorization. A call to a disabled tool family is recorded as a failure
and audited, never dispatched.

`get_latest_document` takes a `source` path, not a topic: the intended
pattern is `search_knowledge_base` to find a source, then
`get_latest_document` to resolve its current version (an old version's
path returns the current version's content).

### Security invariants

- **Structured JSON decisions, not native tool calling.** The decision
  nodes call the existing `LLM.generate(system, user)` with JSON-only
  prompts, parsed and validated in `agent/decisions.py`. Control flow
  doesn't depend on a small local model's native tool-calling.
- **One sanitization path.** Every result entering
  `retrieved_evidence`, from any tool, passes through
  `RetrievalPipeline.sanitize_evidence` (field redaction plus injection
  flagging) inside `graph._execute_tool`, so no tool can skip it.
  Direct-fetch results are tagged `origin="tool_fetched"`.
- **The LLM can't supply authorization.** Tool-arg models
  (`agent/tool_schemas.py`) are `extra="forbid"`, so an auth-shaped key is
  rejected and audited (`agent_tool_argument_rejected`). `auth` is always
  a separate parameter supplied by the driver from state.
- **Server-bounded numbers.** `top_k` is bounded and clamped; document
  fetch sizes aren't LLM-writable at all.
- **Tool output is evidence, not instructions**, as in classic RAG.
- **Kill-switch.** `agent.enabled: false` (default): `POST /agent/query`
  always takes the `classic_rag` route with no extra LLM calls.

### Authorization parity for direct-fetch tools

`get_document`, `get_latest_document`, and `get_related_context` call
`VectorStore` directly, so they first resolve the caller's context through
`RetrievalPipeline.resolve_auth` (applying the authorization kill-switch
and freshness exclusions, as `retrieve()` does) and scope by `dataset_id`.
`_execute_tool` passes the same resolved context to `sanitize_evidence`.
The evidence summary shows each item as
`"[i] chunk_id=<id> source=<path>: <text>"` (no `document_id`,
`tenant_id`, or sensitive-field ids), and `agent_tool_select_v3.yaml`
tells the model to copy `chunk_id` verbatim. Because `source` is
redactable metadata, a filename matching a sensitive-field pattern is
hidden from unauthorized roles, and a later fetch by that redacted value
returns nothing (fail-safe; no current corpus file triggers it).
Covered by `test_agent_tools_authorization_parity.py`,
`test_agent_graph_related_context_chunk_id_exposure.py`, and
`tests/integration/test_agent_tool_tenant_isolation.py` (real Postgres).

### API design

`POST /agent/query` (`api/routers/agent_query.py`) is a separate endpoint,
not a mode flag on `/query`, so `/query` stays the fixed, deterministic
baseline. Both share `api/request_auth.py`
(`build_authorization_context`, `enforce_dos_limits`) and the same
`api/deps.py` singletons. `POST /agent/query/stream` streams progress
events over SSE (see [Observability](#observability)).

### Evaluation

`eval/run_agent_eval.py` (deterministic, no hosted judge): routing
accuracy, `unnecessary_agent_rate`, `tool_selection_accuracy` (expected
tool subset, not exact sequence), tool success rate, average tool calls,
`evidence_sufficiency_accuracy` and `retry_success_rate` (proxied by
`retrieval_attempts >= 2`), max-step termination rate, citation support,
keyword-overlap answer correctness, latency, tokens, and a per-category
breakdown. Gold: `data/eval/agentic_extension_gold.jsonl` (separate from
`techfusion_gold.jsonl`), with additive optional `GoldExample` agent
fields.

### Known limitations (deliberate, not hidden)

- The sufficiency/retry metrics use a coarse proxy; `AgentState` keeps
  only the final sufficiency decision.
- On a retried decision call, only the last attempt's tokens are counted.
- The model rarely selects the specialized local tools over
  `search_knowledge_base` (**AGT-004** in `KNOWN_ISSUES.md`); the tools
  themselves are verified by the tests above.

## MCP Integration

`src/rag/mcp/` is an MCP server (official `mcp` SDK, Streamable HTTP)
mounted inside the `rag-api` process. It exposes up to seven tools:

- Four RAG tools: thin adapters over the same `rag.agent.tools.*`
  functions the agent uses.
- Two business-case read tools, `get_customer_case` and
  `get_case_status`, over `rag.mcp.business`, a small in-memory synthetic
  backend standing in for a separate business system.
- One write action, `update_case_status`.

The in-process agent is also an MCP **client** for the three business
tools (`agent/mcp_client.py`). Its four RAG tools stay direct in-process
calls and never go through MCP.

### Architecture

![Agent and MCP architecture: run_agent dispatches the four local RAG
tools directly in Python to RetrievalPipeline and the knowledge base, and
the business tools through rag.agent.mcp_client, which mints an internal
service JWT and calls the /mcp server over Streamable HTTP; the server
resolves identity, verifies the JWT, and dispatches to
rag.mcp.business.store](assets/agent-mcp-architecture.png)

Reading the diagram against the code: the orchestrator is the custom
bounded loop in `rag.agent.graph`, not LangGraph (LangGraph exists only
on the `experiment/langgraph-agent` branch); the vector store is
Postgres + pgvector only; and an external MCP client (Claude Desktop,
another agent) calls the `/mcp` server directly, the same endpoint the
agent's own MCP client reaches, rather than calling `run_agent`.

`rag.mcp.asgi.build_mcp_asgi_app` wraps the `MCPServer` in a Starlette app
that `rag.api.main` mounts at `mcp.server.mount_path` (default `/mcp`),
sharing the process's existing singletons. `mcp.enabled: false` (default)
means the server is never constructed. Identity is governed by the same
`security.auth` config as the HTTP API, so the two boundaries can't drift.

### Identity: transport-resolved, never a tool argument

Each tool declares
`auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)]`.
`Resolve()` parameters are excluded from the tool's JSON schema, so a
client can't even express an `auth`/`tenant_id`/`roles` argument.
`identity.py`'s `resolve_http_identity` (per call, from request headers)
and `resolve_stdio_identity` (per process, from `MCP_AUTH_TOKEN`) both use
the unmodified `rag.api.auth.verify_jwt`. Note: `rag/mcp/server.py`
intentionally omits `from __future__ import annotations`, because the SDK
evaluates annotations against module globals and the closure-local
resolver isn't there.

### Sanitization and exposure

`server.py:_run_tool` dispatches to the matching `rag.agent.tools.*`
function, re-resolves auth (`pipeline.resolve_auth`, scoped by
`dataset_id`), then calls `pipeline.sanitize_evidence`, the same pattern as
the agent's `_execute_tool`. Results serialize as `McpChunkResult`
(`mcp/schemas.py`), the same exposure level as `/query`'s `SourceItem`.

`_harden_argument_schemas` switches every tool's SDK-generated argument
model to `extra="forbid"` and regenerates its schema
(`additionalProperties: false`), so unknown arguments fail validation
before any tool or resolver runs. (No public SDK hook exists; it edits
`MCPServer`'s internal `ToolManager`.)

`mount_mcp_app` adds a small `_BareMountPathMiddleware` that rewrites an
exact `/mcp` request to `/mcp/`, because Starlette's mount doesn't match
the bare path and the SDK client doesn't follow the 307 redirect.

### Business tools

`get_customer_case`/`get_case_status` use `_run_business_tool` (same
tracing/metrics, no `sanitize_evidence`, since cases have no
chunk-level fields) and receive a bare `VerifiedIdentity`.
`mcp/business/store.py` is an in-memory dict of synthetic cases. Its
authorization mirrors the document ACL rule (own tenant plus a role on
the case, or a `cross_tenant_support_roles` role also listed on the case)
but has no kill-switch, since every case has a tenant. `identity is None`
(auth disabled) means unrestricted for reads. A forbidden case and a
missing case both return `None`; because the check runs in Python, a real
denial is logged as `authorization_denied` with the `case_id`.

### Agent as MCP client

`_execute_tool` routes a tool name in `REMOTE_MCP_TOOL_NAMES` to
`_dispatch_mcp_tool`; local tools use the unchanged `_dispatch_tool`.
Remote calls count toward `max_tool_calls`. The LLM's decision JSON is the
same for both.

- **Fails closed without identity.** `mcp.client.enabled: true` requires
  `security.auth.enabled: true`, checked at startup
  (`validate_startup_config` raises before serving). Each call also
  rejects a missing `AuthorizationContext` or `tenant_id` before any
  request is made.
- **Internal service token, never the caller's JWT.**
  `mint_internal_token` signs a short-lived token
  (`mcp.client.internal_token_ttl_seconds`, default 60) from the
  already-verified `tenant_id`/`roles`, with the shared
  `security.auth.jwt` secret. `sub` is always
  `mcp.client.internal_token_subject` (`rag-agent-internal`). HS256 only
  (the only algorithm with a signing key here); startup refuses other
  algorithms. `iss` is always set; `aud` only when an audience is
  configured, because PyJWT rejects any `aud` when the verifier expects
  none.
- **Transport.** Default `mcp.client.transport: asgi`: an
  `ASGITransport` bound to the same `get_mcp_asgi_app()` singleton the
  mount uses (a second app instance would never have its session-manager
  lifespan entered). It still runs the full Streamable HTTP/JSON-RPC
  protocol, without a socket. The base URL is `http://127.0.0.1:1/` so the
  Host header satisfies the SDK's DNS-rebinding check. `transport: http`
  calls a real `server_url` with no code change.
- **One `ClientSession` per remote call**, bridged with `anyio.run` (safe
  because both callers of `run_agent()` run off the event loop). This
  keeps identities isolated; remote calls are rare relative to LLM
  latency.
- **Synthetic evidence.** Results become `SearchResult(origin="mcp_remote")`
  with `chunk_id = "mcp:{tool}:{case_id}"`,
  `source = "mcp://business/{case_id}"`, and all document-governance
  fields `None`. They never touch `VectorStore`, freshness, or expansion,
  so those can't apply. They still pass `sanitize_evidence` (injection
  flagging on the case description). A synthetic `chunk_id` passed to
  `get_related_context` is an ordinary miss. Reusing `SearchResult` means
  `document_id` holds a `case_id` and `source` a URI.

### `update_case_status`, the write action

Registered only when `mcp.business_actions.enabled: true` (default
`false`), independent of `mcp.client.enabled`.

- **Server-side rules.** Fixed transitions
  `open -> in_progress -> resolved -> closed` (`closed` terminal), and
  `resolved -> closed` requires approval. Same authorization as the read
  tools, but `identity is None` is a hard denial. Requesting the current
  status is a no-op (`already_in_status`), checked before approval. Invalid
  transitions and missing approvals return an outcome without mutating.
  Read-check-write runs under `_CASE_MUTATION_LOCK`.
- **Approval is unreachable from the LLM.** `UpdateCaseStatusArgs` has
  only `case_id` and `new_status`. A `POST /agent/query` caller may send
  `case_approvals` (exact `(case_id, new_status)` pairs), honored only if
  the verified identity holds an `mcp.business_actions.approval_roles`
  role (`request_auth.resolve_case_approvals`), else dropped. The accepted
  list is stored on `AgentState.case_approvals` before the run starts and
  carried as a `case_approvals` claim on the internal token. An approval
  can't apply to a different case or status.
- **Server re-verifies approvals.**
  `business/approvals.resolve_case_action_approvals` honors the claim only
  on a token whose `sub`/`token_use` match `mint_internal_token`'s markers
  and whose embedded `roles` include an approval role. A direct MCP caller
  with an ordinary signed token can't self-approve.
- **One attempt per run.** After dispatching a tool in
  `WRITE_ACTION_TOOL_NAMES`, the tool loop ends regardless of outcome and
  goes to synthesis; `approval_required` is never retried.
  `case_approvals` is size-bounded three times: schema
  (`MAX_CASE_APPROVALS`), router
  (`mcp.business_actions.max_case_approvals_per_request`), and in
  `mint_internal_token`.

### Dependencies

`mcp>=2.1,<3` is a core dependency, since `rag/api/main.py` imports
`rag.mcp.asgi` unconditionally; it's also in the pre-commit mypy hook's
`additional_dependencies`. `httpx2` (the SDK's transport client, used by
tests) is declared in the `dev` extras.

### Testing

- Identity and audit hygiene: `test_mcp_identity.py`.
- Disabled is a true no-op (`/mcp` 404s, server never built):
  `test_mcp_disabled_is_a_true_noop.py`.
- Wire protocol through a real uvicorn server, argument rejection, mount
  and lifespan, business-tool grant/deny matrix:
  `tests/integration/test_mcp_end_to_end.py`.
- Real-Postgres tenant isolation, one scenario per RAG tool:
  `test_mcp_tenant_isolation.py`.
- Agent client, token minting, fail-closed guards:
  `test_mcp_client_stage2.py`, `test_agent_graph_mcp_stage2.py`,
  `tests/integration/test_agent_mcp_client_stage2.py`.
- Write action rules and adversarial approvals:
  `test_mcp_business_case_actions.py`, `test_mcp_business_approvals.py`.

### Known limitations (deliberate, not hidden)

- The rate limiter doesn't cover the MCP mount, and agent calls over the
  in-process ASGI transport bypass the outer app's middleware.
- `_harden_argument_schemas` depends on SDK internals.
- Internal token minting is HS256 only.
- The case store is in-memory and single-process: a fixed seed set with
  no history beyond audit events, a hardcoded transition table, and a
  process-local lock. It is not a production write path, and there's no
  exactly-once guarantee if the process crashes after a mutation.
- Whether the local model's final answer faithfully reports an
  `approval_required` or `invalid_transition` outcome hasn't been
  verified against a real model.

## Observability

Per-node agent latency (split into LLM inference and overhead),
OpenTelemetry traces, Prometheus metrics, a local Grafana dashboard, and a
live progress stream for agent queries. None of it changes agent decision
logic, prompts, or retrieval.

- **OpenTelemetry**: per-request traces.
- **Prometheus**: aggregate operational metrics.
- **Grafana**: dashboards over both (Jaeger datasource for traces).
- **MLflow**: offline experiment tracking, a separate concern.

### Instrumentation points

Node functions in `agent/graph.py` are unmodified; instrumentation lives in
wrappers:

- **`_TimingLLM`** wraps the injected `LLM` and accumulates `total_llm_ms`
  across calls, including JSON-parse retries. Other attributes forward to
  the wrapped instance.
- **`_call_node`** times each node call, opens a span named after the
  node, and splits `llm_ms` from `overhead_ms` using `_TimingLLM`'s delta.
  `execute_tool` reports `None` for both (no LLM call).
- **`_execute_tool`** adds a per-tool span and the tool metrics.
- **`RetrievalPipeline._retrieve_timed`** opens a `retrieval` span and
  records retrieval latency around `_retrieve_timed_inner`; this covers
  both `/query` and the agent's `search_knowledge_base`.
- **`RequestIDMiddleware`** adds the HTTP span and HTTP metrics.

### Node timing

`AgentState.node_timings_ms` maps node name to a list of
`NodeInvocationTiming` (`total_ms`, `llm_ms`, `overhead_ms`), one per
invocation, since retry-loop nodes can run several times.
`node_token_usage` holds per-node token totals. `AgentRunResult` adds
aggregated `node_timings_ms` (`NodeTimingStats`), `llm_call_count`, and
`node_token_usage`. On the agent route, the existing `generation_ms` is the
sum of all agent LLM node durations, not one generation call.

### OpenTelemetry

`observability/tracing.py` is the only module importing the OTel SDK and
OTLP exporter; elsewhere only `opentelemetry.api` is used.
`configure_tracing` installs a real provider only when
`observability.tracing.enabled` (default `false`); otherwise the API's
no-op provider makes every span free. `start_span` guards span creation,
attributes, and teardown so tracing failures never fail a request, while
the caller's own exceptions still propagate.

```
POST /agent/query                    (HTTP root span, RequestIDMiddleware)
  agent_query                        (root span, opened in run_agent())
    classify
    decompose
    tool_select
    tool_execute
      search_knowledge_base          (per-tool-name span)
        retrieval                    (RetrievalPipeline._retrieve_timed)
    evidence_sufficiency
    ... (loop repeats on insufficient evidence)
    synthesize
```

Span attributes include route, tool name and success, step count,
sufficiency/termination, and per-node timings. Never attached: JWTs,
credentials, chunk text, raw prompts, or the decision models' `reasoning`
field. `JSONFormatter` adds `trace_id`/`span_id` to log records when a span
is active.

### Prometheus

`observability/metrics.py` uses its own `CollectorRegistry` (avoids
duplicate-timeseries errors on re-import), and every `observe_*`/`inc_*` is
wrapped to log rather than raise. Labels are bounded: no query text,
tenant, document, chunk id, or tool arguments. `path` is the matched route
template.

| Metric | Type | Labels |
|---|---|---|
| `rag_http_requests_total` | Counter | `method`, `path`, `status_code` |
| `rag_http_request_duration_seconds` | Histogram | `method`, `path` |
| `rag_agent_requests_total` | Counter | `route` (`classic_rag`/`agent`) |
| `rag_agent_total_latency_seconds` | Histogram | `route` |
| `rag_agent_steps` | Histogram | none |
| `rag_agent_tool_calls_total` | Counter | `tool_name`, `success` |
| `rag_agent_tool_latency_seconds` | Histogram | `tool_name` |
| `rag_agent_node_latency_seconds` | Histogram | `node` (total time) |
| `rag_agent_node_llm_latency_seconds` | Histogram | `node` (LLM-inference-only time) |
| `rag_retrieval_latency_seconds` | Histogram | `provider` |
| `rag_agent_termination_reason_total` | Counter | `reason` |
| `rag_agent_evidence_sufficiency_total` | Counter | `sufficient` |
| `rag_errors_total` | Counter | `component` |

The same registry also holds `rag_context_*`, `rag_graph_*`/`rag_graphrag_*`,
`rag_feedback_*`, and `rag_authorization_denials_total` metrics from those
subsystems.

### `/metrics`, `GET /`, and `GET /info`

`GET /metrics` is always registered and returns 404 unless
`observability.metrics.enabled`. `GET /` is dependency-light: service
status, links, and a `FeatureFlags` block of booleans and provider names
(never a model name, host, or connection string). `GET /info` returns
`RuntimeInfo`: concrete provider/model names, retrieval mode, fusion,
reranker, and MCP toggles. It's a separate endpoint so `GET /` can keep
its no-model-name guarantee. Neither exposes JWTs, keys, DB URLs, paths,
tokens, or prompt text. The web UI shows `/info` in a collapsible panel.

### Live agent events (SSE)

`POST /agent/query/stream` (`api/routers/agent_stream.py`) streams
`AgentEvent`s (`agent/events.py`) and ends with one
`completed`/`terminated` event shaped like `/agent/query`'s response.
`AgentEvent` has no free-text field (only event type, step, tool name,
elapsed time, chunk count, sufficiency, termination reason, route), so
there's nowhere for prompts, chunk text, reasoning, or credentials to go.
SSE fits one-way progress; POST keeps the same request body as
`/agent/query`.

`run_agent` runs in Starlette's threadpool, bridged to the event loop via
an `asyncio.Queue`. On client disconnect (polled while the queue is idle),
the router sets a `cancel_event` that `run_agent` checks between loop
iterations, ending with `termination_reason="cancelled"` and no synthesis.
A node call already in progress (e.g. an LLM call) is not interrupted.
Gated by `observability.live_events.enabled` (default `true`; 404 when
off).

### Local Grafana / Prometheus / Jaeger

`docker-compose.observability.yml` (`make observability-up`), never part
of plain `make up`:

- **Jaeger** (all-in-one): accepts OTLP directly on 4317/4318; UI on
  `:16686`.
- **Prometheus**: scrapes `rag-api:8000/metrics` every 15s; `:9090`.
- **Grafana**: provisioned datasources plus the
  `agentic-rag-overview.json` dashboard on `:3000` (anonymous viewer, local
  dev only). Panels cover success/error rate, route latency, per-node
  total vs. LLM latency, steps and tool calls per run, slow tools,
  termination and sufficiency breakdowns, and errors by component.

Starting the stack doesn't enable tracing; set
`observability.tracing.enabled: true`.

### Known limitations (deliberate, not hidden)

- Instrumentation overhead was only sanity-checked (mocked-LLM runs stay
  under 50ms mean); there is no telemetry-off baseline, since metric
  recording is unconditional.
- Disconnect cancellation is cooperative; a node call already running
  finishes first.
- Reused threadpool threads can log a noisy but harmless OTel "Failed to
  detach context" error (see `KNOWN_ISSUES.md`).
- HTTP spans are hand-written; no OTel auto-instrumentation package.

## Layout-Aware Document Ingestion and Vision

PDF and DOCX ingestion preserve structure (headings, tables, code/config
blocks, images, page numbers, captions) instead of flattening to text, and
`OllamaVisionProvider` adds local, offline image descriptions.

### Design: serialize into the existing Markdown grammar

`PDFLoader`/`DocxLoader` keep the `Loader.load(path) -> RawDocument`
interface and emit Markdown-equivalent `content` (`#`/`##` headings, pipe
tables, fenced code blocks, `![alt](path)` image lines) that
`structured_markdown.py` already parses. The chunker's only change is
accepting `source_type in {"markdown", "pdf", "docx"}`. So PDF/DOCX chunks
have the same shape as Markdown chunks, and redaction, injection flagging,
relationship expansion, and authorization need no changes. (A separate
typed element model was rejected because it would duplicate that logic.)

The one new syntax is a `<!--page:N-->` sentinel per PDF page or DOCX
manual page break. The chunker strips it and sets `ChunkMetadata.page`
until the next marker; Markdown/text/HTML keep `page=None`.

### PDF extraction (`pdfplumber`)

`ingestion.layout_parsing.pdf_parser: pdfplumber` (the only option; chosen
over PyMuPDF for licensing). `pypdf` is still used for document metadata
and as the raw image-byte source.

`PDFLoader._render_page` merges paragraphs, tables
(`page.find_tables()`), and images by vertical position into reading
order:

- **Headings** are a font-size heuristic: lines at
  `>= heading_font_size_ratio` (1.15) times body size and at most
  `max_heading_words` (12) words become `##`; above
  `title_font_size_ratio` (1.8), `#`. Deeper nesting collapses to two
  levels.
- **Code/config**: lines entirely in a monospace font become fenced
  blocks; `markdown_render.sniff_code_language` tags `json`/`xml`/`yaml`
  by shape, else leaves the fence untagged.
- **Captions**: a paragraph matching `^(figure|table)\s+\d+[.:)]` right
  after an image is emphasis-wrapped so the chunker's existing caption
  rule folds it into the image span.

### DOCX extraction

`DocxLoader._render_body` walks the body's paragraphs and tables in order:

- **Pages**: only explicit manual page breaks (`<w:br w:type="page"/>`)
  are visible; Word's reflow pagination isn't computable, so a DOCX with
  no manual breaks is `page=1` throughout.
- **Images**: `.//a:blip` per paragraph, bytes via `related_parts`.
- **Code/config**: paragraphs whose runs are all monospace
  (`_is_monospace_paragraph`); manual line breaks are preserved.
- **Headings**: `Title` becomes `#`, `Heading N` becomes `##` (same
  two-level depth as PDF).

### Image-asset resolution

`loaders/base.py:resolve_image_asset` prefers a pre-supplied sibling
`assets/` folder. When several documents share one folder,
`_matching_assets` narrows candidates to files sharing a non-stopword
filename word with the document (e.g. `ingestion-capacity-analysis.docx`
and `capacity-view-01.png`), falling back to all candidates in sorted
order (MULTI-001). With no `assets/` folder, the image's own bytes are
written as `<stem>-figure-{n:02d}<ext>` (deterministic, idempotent on
re-ingest).

### Vision: `OllamaVisionProvider`

`vision.provider: none | ollama` (default `none`). Calls Ollama's `chat()`
with image bytes; default model `moondream` (~1.7GB, suits the 8GB CPU-only
host), pulled manually by the operator. It uses the same local Ollama
server as generation, so it isn't a hosted egress point.

`image_description_cache` is keyed by
`(image_checksum, provider, model_name, prompt_version)`, so switching
model or prompt never reuses another model's description; each
`VisionProvider` declares `prompt_version`. `Writer._with_vision_siblings`
calls the provider only on a cache miss, inside a try/except: a failed call
logs `vision_provider_call_failed` and skips that one image without failing
the document (MULTI-003). Stats (`VisionCallStats`: images, hits/misses,
latency) are reported via `IngestionStats.vision_stats`.

A real A/B/C run (text-only vs. layout-aware vs. layout-aware plus vision)
showed vision descriptions retrieved and cited correctly, but no
answer-quality gain on vision-required questions, because `moondream`'s
descriptions are weak (MULTI-002, an accepted limitation).

### Agent `content_type` filter

`SearchKnowledgeBaseArgs.content_type` is an optional closed `Literal`
(`prose`, `table`, `code`, `configuration`, `image`, `chart`). It's merged
into the server-controlled `filters` (validated against
`ALLOWED_FILTER_FIELDS`); the LLM can pick a value but can't add, remove,
or override other filter keys such as `dataset_id`.

### API metadata

`SourceItem` and `agent.state.Citation` include `content_type`,
`section_path`, `page`, `attachment_name`, `source_anchor`, and
`vision_generated`: non-sensitive structural metadata with relative,
dataset-scoped paths only. Prompt source labels include `page N` so the
model can cite pages.

### Deterministic metrics (`layout_vision` section of `eval/run_eval.py`)

From the same top-10 retrieval as Recall@10:

- `table_retrieval_hit_rate` / `visual_retrieval_hit_rate`: a chunk of the
  expected `content_type` from a relevant document was retrieved.
- `page_localization_accuracy`: a retrieved relevant chunk is on one of
  the gold `relevant_pages` (PDF/DOCX only by construction).
- `section_localization_accuracy`: gold section label is a substring of
  the chunk's `section_path` (heuristic).
- `visual_evidence_support_rate`: a `vision_generated` chunk from a
  relevant document was retrieved; always 0 with vision off.
- `multimodal_answer_quality` (generation runs): keyword-overlap answer
  quality over the vision/layout/expansion-dependent subset.

Gold extensions (`requires_layout_awareness`, `expected_content_types`,
`relevant_pages`, `visual_question_type`, `evidence_mode`, `source_format`)
are optional. Gold set: `techfusion_layout_vision_extension_gold.jsonl`
(28 questions), corpus under
`data/knowledge_base/layout_vision_extension/`. Experiment records and
MLflow log `layout_parser`, `vision_provider`, `vision_model`, and
`vision_prompt_version`.

### Known limitations (deliberate, not hidden)

- PDF headings are a font-size heuristic, flattened to two levels.
- DOCX page numbers need explicit manual page breaks.
- PDF image-index pairing between `pdfplumber` and `pypdf` assumes the
  same stream order; verified on simple single-column documents only.
- With a vision provider configured, the change-detection checksum also
  covers every referenced local image, so swapping or removing an image
  re-ingests its document; with vision off, image bytes feed nothing and
  are ignored.

## Feedback loop (feedback milestone)

A caller's rating of an answer becomes a durable, tenant-isolated,
reviewable record without turning feedback into a new sensitive-data sink.
Scope is persistence and export; promotion into gold data stays manual.

```
answer (POST /query or /agent/query*)
  -> caller sees answer + request_id
  -> thumbs up/down (+ optional structured reason/comment)
  -> POST /feedback
  -> tenant-scoped Postgres row (rag.feedback.store.FeedbackStore)
  -> scripts/export_feedback.py (JSONL/CSV, filtered)
  -> human review
  -> optional, deliberate, hand-authored promotion into a gold eval file
```

### `request_id`

`QueryResponse`/`AgentQueryResponse` include `request_id`, the same
per-request id used for the `x-request-id` header and log lines. The SSE
route captures it synchronously before building the stream (the contextvar
may be reset before the generator's terminal frame runs). No server-side
answer log exists, so `request_id` is an opaque, length-bounded (1-100)
caller-asserted correlation string; an unknown id is still accepted.

### API and trust

`POST /feedback` (`api/routers/feedback.py`) uses the same identity rules as
`/query`: `request_auth.resolve_trusted_tenant_id` takes `tenant_id` from a
verified JWT over the body (mismatch logged as `forged_claim_attempt`), and
`get_current_identity` gives the same 401 behavior. `FeedbackRequest` is
`extra="forbid"`. Reasons come from closed vocabularies
(`NEGATIVE_FEEDBACK_REASONS`/`POSITIVE_FEEDBACK_REASONS`), and a reason from
the wrong rating's set is a 422. The route returns 404 when feedback is
disabled.

### Storage and dedup

`feedback/store.py:FeedbackStore` has its own small connection pool (not
`PgVectorStore`'s); schema comes from `scripts/init_db.py`. One row per
`(tenant_id, caller_key, request_id)`, enforced by a `UNIQUE` constraint
and a single `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` (atomic under
concurrency); a resubmission updates the rating, reason, and comment while
keeping `feedback_id`.

- `tenant_id` is `TEXT NOT NULL DEFAULT ''` (read back as `None`), because
  Postgres treats NULLs as distinct in `UNIQUE` keys, which would break
  dedup for anonymous submissions.
- `caller_key` is the pseudonymous hash of the verified `sub`, or
  `"anonymous"`. Two unauthenticated callers on the same `request_id`
  share a row; distinct callers need verified identity.

Cited sources and tool names are `TEXT[]` columns; lineage fields are
normalized columns. Indexes cover `tenant_id`, `created_at`, `rating`,
`dataset_id`, and `route`.

Lineage (`generation_model`, `prompt_id`/`prompt_version`,
`retrieval_provider`, `reranker_provider`) is read from config when
feedback is **submitted**, not when the answer was generated. That's exact
while config only changes on restart, but it's a snapshot, not a
per-answer record. The stored `query_text`/`answer_text` are the reliable
reference.

### Privacy and security

- No JWT, token, or `AuthorizationContext` reaches the store; it receives
  only plain values (`test_no_jwt_or_auth_context_ever_reaches_the_store`).
- `tool_calls` holds tool names only, never arguments, reasoning, or
  intermediate LLM output.
- Stored query and answer text are what the caller already had
  (already redacted at retrieval time), length-bounded, and each
  toggleable off (`store_query_text`, `store_answer_text`).
- `comment` and list sizes are bounded from runtime config
  (`_enforce_feedback_limits`). Comment contents aren't scanned, like any
  other caller-authored free text.
- Audit event `feedback_submitted` carries only tenant, rating, route, a
  has-comment boolean, and outcome.

### Export and gold boundary

There's no public `GET /feedback`. `scripts/export_feedback.py`
(`FeedbackStore.export`, filtered by tenant, rating, dataset, route, and
date) writes JSONL/CSV for human review and writes nowhere else
(`test_write_jsonl_never_mutates_a_gold_dataset_file`). Promoting feedback
into a gold file is a separate, manual step.

### Observability and frontend

`rag_feedback_submissions_total` (`rating`, `outcome`:
`created`/`updated`/`failed`) and `rag_feedback_submission_latency_seconds`;
storage failures also count in `rag_errors_total{component="feedback"}`.
`frontend/.../FeedbackControls.tsx` shows thumbs up/down plus an optional
reason/comment, only for messages with a `request_id`. Re-clicking the
selected rating is a no-op; switching rating sends an update.

### Out of scope

Automatic feedback-to-gold promotion, feedback-driven CI gates, judging
feedback-flagged examples with RAGAS, a feedback queue, and a
feedback-browsing endpoint.
