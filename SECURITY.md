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

All of the following are off by default (`config.security.*` in
`config/default.yaml`) and independently toggleable, so the base RAG
pipeline works unmodified without them. See `docs/architecture.md`'s
"Authorization, Freshness, and Trust", "Field-Level Sensitive-Data
Redaction", and "Authenticated API Boundary and Security Hardening"
sections for the full design and documented limitations of each.

- **Authenticated API boundary**: JWT verification (`HS256`/`RS256`/
  `ES256`) at the API boundary, with signature/expiration/issuer/audience
  and required-claim checks (`src/rag/api/auth.py`).
- **Retrieval-time authorization**: tenant and role-based access control
  enforced as SQL predicates in the vector store, before any row leaves
  Postgres (`src/rag/retrieval/authorization.py`).
- **Ingest-time tenant governance**: an authenticated `POST /ingest`
  upload is always bound to the caller's own verified tenant (stamped when
  the uploaded document carries no governance metadata of its own, e.g.
  every PDF/DOCX upload; rejected if it explicitly names a different
  tenant, unless the caller holds a cross-tenant support role), so an
  authenticated upload can never silently become globally visible across
  tenants (`src/rag/ingestion/governance.py`).
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

## Known limitations

These are documented, not hidden, in `docs/architecture.md`:

- No authentication layer existed prior to the auth-boundary milestone;
  `tenant_id`/`roles` are trusted claims once verified, not independently
  re-checked against an external identity provider.
- Prompt-injection detection is a small literal/regex heuristic, not a
  robust classifier.
- Rate limiting uses in-memory state, so it is process-local and does not
  aggregate across multiple API replicas.
