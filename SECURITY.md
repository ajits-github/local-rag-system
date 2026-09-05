# Security Policy

## Scope

This is a personal, portfolio-oriented RAG system. It runs offline against
synthetic sample/eval data, is not deployed as a public service, and has no
real user data. The security controls described below are implemented and
tested as part of the project's engineering scope, not as guarantees for a
production deployment.

## Reporting a vulnerability

If you find a security issue in this repository (e.g. an authorization
bypass, an injection vector, a redaction gap), please report it privately
rather than opening a public issue: email ajit110920@gmail.com with a
description and, if possible, steps to reproduce. Expect an acknowledgement
within a few days; this is a solo-maintained project, not a monitored
security inbox with an SLA.

## Implemented security controls

All of the following are off by default (`config.security.*`/`config.agent.*`/
`config.mcp.*` in `config/default.yaml`) and independently toggleable, so
the base RAG pipeline works unmodified without them. See
`docs/architecture.md`'s "Authorization, Freshness, and Trust",
"Field-Level Sensitive-Data Redaction", "Authenticated API Boundary and
Security Hardening", "Agentic RAG", and "MCP Integration" sections for
the full design and documented limitations of each.

- **Authenticated API boundary**: JWT verification (`HS256`/`RS256`/
  `ES256`) at the API boundary, with signature/expiration/issuer/audience
  and required-claim checks (`src/rag/api/auth.py`).
- **Retrieval-time authorization**: tenant and role-based access control
  enforced as SQL predicates in the vector store, before any row leaves
  Postgres (`src/rag/retrieval/authorization.py`).
- **Ingest-time tenant governance**: an authenticated `POST /ingest`
  upload is always bound to the caller's own verified tenant. The system
  stamps the caller's tenant when the uploaded document carries no
  governance metadata of its own, such as every PDF/DOCX upload, and
  rejects documents that explicitly name a different tenant unless the
  caller holds a cross-tenant support role. This prevents authenticated
  uploads from silently becoming globally visible across tenants
  (`src/rag/ingestion/governance.py`). Upload persistence itself is
  atomic: an upload is staged under a random directory, keeping its own
  original filename, and only installed via an atomic rename once size
  validation, parsing, and governance all succeed. A rejected or oversized
  re-upload can never truncate or destroy a previously accepted file under
  the same name, and no staging-directory fragment can leak into a
  document's title, extracted-image filenames, or citation metadata
  (`src/rag/api/routers/ingest.py`).
- **Document-version freshness**: superseded/stale document versions are
  resolved and excluded per query (`src/rag/retrieval/freshness.py`).
- **Field-level redaction**: specific sensitive fields (e.g. credentials)
  within an otherwise-authorized document are redacted from retrieved
  content and metadata before reaching the prompt
  (`src/rag/retrieval/field_policy.py`).
- **Prompt-injection detection**: retrieved content is flagged when it
  contains embedded-instruction patterns; this is telemetry and prompt
  reinforcement, not a hard gate (`src/rag/retrieval/injection_detection.py`).
- **DoS limits and rate limiting**: request size/query-length/top-k
  bounds and per-tenant rate limiting at the API layer
  (`src/rag/api/routers/query.py`, `src/rag/api/deps.py`).
- **Audit logging**: authorization/authentication/redaction decisions are
  logged as structured events, with the JWT subject claim pseudonymized
  before logging, never raw (`src/rag/audit.py`).
- **Agentic tool-calling boundaries**: every agent tool's argument schema
  rejects unknown fields (`extra="forbid"`), so the LLM can never supply
  or override identity, and every LLM-writable numeric argument is
  server-side range-clamped. Every tool's output passes through the same
  evidence-sanitization step (field redaction, injection flagging)
  regardless of which tool produced it. The tool-calling loop is bounded
  by independent step/retrieval/tool-call limits, not a single blended
  recursion budget (`src/rag/agent/`).
- **MCP integration**: identity is resolved from the transport (a
  verified JWT) and injected server-side; it is never a tool argument a
  client can supply. Every tool's argument schema is hardened to
  `extra="forbid"` after registration, and every result passes through
  the same evidence-sanitization step the in-process agent uses
  (`src/rag/mcp/`). The agent-as-MCP-client path mints a fresh,
  short-lived internal service token per call from the caller's already-
  verified identity, never forwarding the caller's own JWT, and fails
  closed at startup unless authentication is enabled. The one write
  action (`update_case_status`) enforces a fixed transition table and
  requires an explicit, role-gated approval for its one sensitive
  transition; that approval is honored only from a token that is
  unambiguously agent-minted (`sub`/`token_use` match, plus an
  independent re-check of the token's own embedded roles), closing a
  real authorization-bypass gap found by post-implementation review (see
  `docs/architecture.md`'s "MCP Integration" section, "Stage 3").
- **Provider-egress policy**: the one hosted-LLM call site in this
  codebase (RAGAS judge scoring) is gated on tenant/classification/
  trust-level and unredacted-sensitive-field checks before any content
  leaves the process; production `answer()` never calls a hosted LLM at
  all (`src/rag/eval/egress_policy.py`).

## Known limitations

These are documented, not hidden, in `docs/architecture.md`:

- No authentication layer existed prior to the auth-boundary milestone;
  `tenant_id`/`roles` are trusted claims once verified, not independently
  re-checked against an external identity provider.
- Prompt-injection detection is a small literal/regex heuristic, not a
  robust classifier.
- Rate limiting uses in-memory state, so it is process-local and does not
  aggregate across multiple API replicas.
- The rate limiter does not wrap the MCP mount; MCP requests are not
  rate-limited today.
