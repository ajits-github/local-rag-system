# Authorization call sites

A caller's request can be checked at up to four independent layers:
API-level authentication, retrieval-level document authorization,
field-level redaction, and (for the agent's one write action)
tool/action-level permission. These are deliberately separate,
composable controls, not one blended check, but within each layer the
same function is reused everywhere that layer applies, never
reimplemented per caller. This page lists the exact function and every
call site, verified directly against the code rather than assumed.

## 1. API-level: can this user call this endpoint at all?

`api/auth.py`'s `verify_jwt` (decodes and verifies the bearer token),
wrapped by the FastAPI dependency `api/deps.py:get_current_identity`.
Every route that needs an identity (`/query`, `/ingest`, `/agent/query`,
`/agent/query/stream`, `/feedback`) takes this same dependency. For
`/ingest`, this is also where the `ingest_roles` role check happens.

## 2. Retrieval-level: which documents/chunks may this user see?

- `RetrievalPipeline.resolve_auth` decides whether the
  `config.security.authorization.enabled` kill-switch is on and
  resolves freshness exclusions. Called from:
  - `retrieval/pipeline.py`'s `retrieve()` and `retrieve_attribution()`.
  - `agent/graph.py`'s `_execute_tool` (the universal per-tool-dispatch
    step every agent tool call passes through).
  - `agent/tools.py`'s `get_document`/`get_latest_document`/
    `get_related_context` (the three direct-fetch tools).
  - `mcp/server.py`'s `_run_tool`.
- `vectorstore/pgvector.py`'s `build_authorization_where_clause` is the
  actual SQL predicate, called once per read method inside
  `pgvector.py` (`search`, `search_keyword`, `get_chunks_by_ids`,
  `get_chunks_by_section`, etc.), so the tenant/role/trust rule is
  identical regardless of which path reaches the database.

This consistency was not always true. `KNOWN_ISSUES.md`'s "Agent tools that
queried the vector store directly skipped the authorization
kill-switch" entry documents a real bug: the three direct-fetch tools
originally bypassed `resolve_auth` entirely, since they called
`VectorStore` directly instead of going through `retrieve()`. Fixed by
threading `resolve_auth` through them explicitly, which is why it now
appears in `agent/tools.py` as well as `retrieval/pipeline.py`.

## 3. Field-level: even an allowed document may contain a sensitive value

`RetrievalPipeline._redact_sensitive_fields` (internally calling
`field_policy.redact_sensitive_fields(text, roles)`) is the one choke
point, called from five places inside `_retrieve_timed` covering dense,
BM25, and fused results, so no retrieval path can skip it. For
tool-fetched evidence that never went through `retrieve()`, the public
wrapper `RetrievalPipeline.sanitize_evidence` does the same redaction,
called from `agent/graph.py`, `mcp/server.py`, and `rag.graph.pipeline`.
Same underlying logic, two entry points for two different evidence
shapes.

## 4. Tool/action-level: does this user have permission to do *this*?

Deliberately **not** the same mechanism as the three layers above. A
customer case has no document/freshness/trust concept, so it never
reuses `AuthorizationContext`. Instead:

- `mcp/business/store.py`'s `_lookup_authorized` (built on
  `_is_authorized`) is the shared check, called by all three business
  tools: `get_customer_case`, `get_case_status`, and
  `update_case_status`. Same tenant/role predicate every time.
- `update_case_status` additionally layers
  `mcp/business/approvals.py`'s `resolve_case_action_approvals` on top,
  since a mutation needs an explicit, role-gated approval that a read
  does not.

## Summary

Authentication and document/field authorization are each centralized
into one real shared function per layer, verified by call site, not
duplicated per caller. Tool/action-level authorization for the one
write action is intentionally a separate, parallel system, not an
oversight, since the resource it protects is not a document at all.
