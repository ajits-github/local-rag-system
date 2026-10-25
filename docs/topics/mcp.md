# MCP

--8<-- "README.md:docs-mcp"

Every existing security control this codebase already built (JWT
identity, tenant/role authorization, freshness, trust filtering,
field-level redaction, prompt-injection detection, audit logging) stays
active on the MCP path and cannot be bypassed by an MCP tool call, the
same guarantee [Agentic RAG](agentic-rag.md) makes for the in-process
agent; see [Security](security.md) for the control-by-control breakdown.

## The four RAG tools

| Tool | What it does | Underlying function |
|---|---|---|
| `search_knowledge_base` | Query the authorized knowledge base | `rag.agent.tools.search_knowledge_base` |
| `get_document` | Fetch a specific document by source path, bounded and relevance-selected | `rag.agent.tools.get_document` |
| `get_latest_document` | Resolve a (possibly superseded) source to its current version, then fetch it | `rag.agent.tools.get_latest_document` |
| `get_related_context` | Fetch parent/neighbor context for an already-retrieved chunk, by `chunk_id` | `rag.agent.tools.get_related_context` |

None of these reimplement retrieval or authorization logic: they are
thin MCP adapters over the exact same, already-adversarially-tested
functions `rag/agent/graph.py`'s in-process tool dispatch calls.

## Synthetic business-backend tools (Stage 1B read tools, Stage 3 write action)

| Tool | What it does | Underlying function |
|---|---|---|
| `get_customer_case` | Fetch a synthetic customer-support case's full detail by `case_id` | `rag.mcp.business.store.get_customer_case` |
| `get_case_status` | Fetch only a case's status, priority, and last-updated time (a narrower read) | `rag.mcp.business.store.get_case_status` |
| `update_case_status` | Request a status change for a case, subject to a fixed transition table and an approval requirement for one sensitive transition | `rag.mcp.business.store.update_case_status` |

These exist to demonstrate MCP as an integration layer to a separate
backend/business system, not another transport for this
deployment's own RAG tools. `rag.mcp.business.store` is a small,
in-memory synthetic case dataset (no Postgres/network dependency, a
stand-in for a real backend a production MCP server might front) with
its own tenant/role authorization: a case's owning tenant plus a matching
role on the case, or a `security.authorization.cross_tenant_support_roles`
role that is also listed on that case (the same rule, and the same config
list, document retrieval already uses for cross-tenant access, not a
second parallel privilege list). A case the caller may not access is
returned as `null`, identical to a case that doesn't exist. Unlike
document-level ACL (a SQL predicate with no visibility into what it
excluded), this in-Python check *can* observe a real denial, so it logs
an `authorization_denied` audit event even though the response itself
never reveals whether the case exists. `update_case_status` additionally
treats a missing identity as a hard denial (unlike the two read tools'
"no identity, fully unrestricted" convention), since a mutating action
has no legacy unauthenticated-access state to preserve.

## Identity: transport-resolved, never a tool argument

A verified JWT's identity is resolved once per call from the
Streamable HTTP request's own headers (`resolve_http_identity`), or once
per process from the `MCP_AUTH_TOKEN` environment variable for the
stdio transport (`resolve_stdio_identity`); both reuse the same
`verify_jwt` the HTTP `/query` boundary already uses. The MCP SDK's
`Resolve()` parameter-injection mechanism statically excludes the
resolved `auth` value from every tool's generated JSON schema, so a
client cannot supply or override `tenant_id`/`roles`/`auth` even in
principle, not only by convention. Governed entirely by the existing
`security.auth` config tree; there is no separate, weaker `mcp.auth.*`
toggle that could drift from it.

## Two hardening fixes important to know about

- **Unknown tool arguments are rejected loudly**, not silently dropped.
  The MCP SDK's own generated schema doesn't set
  `additionalProperties: false` by default; this codebase hardens every
  tool's argument model to `extra="forbid"` after registration, so an
  injected or arbitrary unknown field fails validation before the tool
  function ever runs.
- **The bare mount path (`/mcp`, no trailing slash) works reliably.**
  Starlette's `Mount` route only matches `<mount_path>/...`, so the bare
  path used to 307-redirect to the trailing-slash form, a redirect the
  MCP SDK's own client doesn't follow during session init. A small
  server-side ASGI middleware now makes both spellings work identically.

## Stage 2: the agent as an MCP client

Stages 1A/1B are server-only: this codebase exposes tools, it doesn't
call any as a client. Stage 2 closes that other half, but only for the
two business tools: `rag/agent/mcp_client.py`, dispatched from
`rag/agent/graph.py`'s bounded tool loop. The four RAG tools stay exactly
what they already were: direct, in-process function calls, never routed
through MCP.

- **Fails closed, not open.** `mcp.client.enabled=True` requires
  `security.auth.enabled=True`, checked once at process startup (not
  per request), since the business tools' authorization has no
  kill-switch to fall back on.
- **Never forwards the caller's own token.** Each remote call mints a
  fresh, short-lived internal service token from the caller's already-
  verified tenant/roles (`sub` is always a fixed, clearly synthetic
  marker), signed with the same secret the receiving, unmodified
  `verify_jwt` already checks against.
- **In-process by default, real network as an escape hatch.** The
  default transport binds an ASGI transport directly to the same MCP
  server object this process mounts (no socket, still the full
  Streamable HTTP protocol end to end). Pointing it at a genuinely
  separate deployment later is a config change, not new code.
- **One session per call**, never pooled across requests: the simplest
  correct answer to "does this call need an already-running event loop,"
  since `run_agent()`'s node functions are synchronous by design.
- **Business-case evidence is synthetic and inert to document logic.**
  A fabricated `chunk_id`/`source`, `origin="mcp_remote"`, and every
  freshness/relationship/ACL-relevant field left unset: this evidence
  never touches `VectorStore` or the retrieval pipeline, so those
  subsystems structurally never see it.

Full design writeup, covering token-minting's `iss`/`aud` asymmetry (a real
PyJWT behavior found mid-implementation), the DNS-rebinding Host-header
bug the ASGI transport hit, and the session-lifecycle tradeoffs:
[MCP Integration](../architecture.md#mcp-integration).

## Stage 3: `update_case_status`, the first write action

The one mutating tool in this codebase. Deliberately narrow (one
transition table, one sensitive transition) rather than a general
write framework, since the point was proving the write-action *shape* is
sound, not building out a full case-management surface.

- **Business rules are deterministic and server-side.** A fixed
  transition table (`open -> in_progress -> resolved -> closed`, `closed`
  terminal); an already-in-target-status request is a defined no-op; an
  invalid transition or a missing approval both leave the case untouched.
- **The LLM cannot supply or forge approval.** `UpdateCaseStatusArgs` has
  only `case_id`/`new_status`; no approval field exists on the schema
  at all. A human caller's `case_approvals` field on `POST /agent/query`
  is honored only when their verified identity holds an approval role,
  then carried as a claim on the same signed internal service token
  Stage 2 already mints, reusing the existing trust channel rather than
  adding a second one. Because each approval names an exact
  `(case_id, new_status)` pair, one approval can never authorize a
  different case or a different transition.
- **A verifiable token alone was not enough to grant an approval: a
  real bypass found by post-implementation review, not the original
  design or test suite.** The approval resolver originally checked only
  that a token verified under the shared JWT config, never that it was
  minted by the agent's own internal-token flow, so a direct
  MCP caller with any ordinary, validly-signed token could attach the
  claim itself and skip the API-boundary role gate entirely. Fixed by
  requiring the token's `sub`/`token_use` to unambiguously mark it as
  agent-minted, plus an independent re-check of that token's own `roles`
  against the approval-role list. Full diagnosis in `KNOWN_ISSUES.md`; full
  design in [MCP Integration](../architecture.md#mcp-integration).
- **A write-action attempt is always the last tool call of a run.**
  Whatever the outcome (executed, needs approval, invalid, already
  done, or a dispatch failure), the agent proceeds straight to synthesis
  rather than retrying the same mutation.

## Config

```yaml
mcp:
  enabled: false      # true no-op when false: the MCP server is never built or mounted
  server:
    mount_path: /mcp
  client:
    enabled: false    # true no-op when false: the agent never offers or dispatches the remote tools
    transport: asgi   # or "http", for a genuinely separate future deployment
  business_actions:
    enabled: false    # true no-op when false: update_case_status is not registered on the server at all
```

`mcp.enabled: false`, `mcp.client.enabled: false`, and
`mcp.business_actions.enabled: false` are all shipped defaults. When
`mcp.enabled` is `false`, the MCP server (and its Streamable HTTP
session manager) is never even constructed, not merely built and left
unmounted; `GET /mcp` 404s exactly like a route that was never
registered. When `mcp.client.enabled` is `false`, the agent's tool-select
prompt never offers `get_customer_case`/`get_case_status`/`update_case_
status`, and a decision naming one anyway fails closed as an ordinary
recorded tool failure, never a real dispatch attempt. `mcp.business_
actions.enabled` is a separate, independent kill-switch specifically for
the write action: it can stay off even when `mcp.client.enabled` is on,
leaving the two read tools available without exposing the write tool.

Full design writeup, including both server-side hardening fixes' root
causes, the alternatives considered and rejected for the bare-mount-path
fix, and the full Stage 2 design:
[MCP Integration](../architecture.md#mcp-integration).

API reference: [MCP](../reference/mcp.md).

## MCP integration (mcp milestone, Stages 1A, 1B, 2, and 3)

A secure MCP (Model Context Protocol) server, `src/rag/mcp/`, exposing up
to seven tools over the official `mcp` Python SDK's Streamable HTTP
transport, mounted into the existing `rag-api` process: the same four
`rag.agent.tools.*` functions the in-process agent already calls
(`search_knowledge_base`/`get_document`/`get_latest_document`/
`get_related_context`, Stage 1A), two synthetic business-case read tools
(`get_customer_case`/`get_case_status`, Stage 1B, `src/rag/mcp/business/`)
demonstrating MCP as an integration layer to a separate backend system,
not just another transport for this deployment's own RAG tools, and one
write action (`update_case_status`, Stage 3, gated by its own
`config.mcp.business_actions.enabled` kill-switch, independent of
`mcp.client.enabled`). Stage 2 makes the in-process agent itself an MCP
*client* for the business tools, the four local RAG tools stay direct,
in-process calls, never routed through MCP. See "Stage 3: the first
write action" below for the full write-action design.

- **`config.mcp.enabled`** (default `false`, `config/default.yaml`) is a
  true no-op, same convention as `agent.enabled`/`security.authorization.
  enabled`: when `false`, `rag/api/main.py` never builds or mounts
  anything, so `/mcp` 404s exactly like a route that was never
  registered. No separate `mcp.auth.*` block exists, identity is
  governed entirely by the existing `security.auth` tree, so the HTTP
  API boundary and the MCP boundary can never drift into different
  security postures. `config.mcp.server.mount_path` (default `/mcp`) is
  the only other field.
- **Identity is transport-resolved, never a tool argument.** Every tool
  declares `auth: Annotated[AuthorizationContext | None,
  Resolve(_resolve_auth)]`, the SDK's `Resolve(fn)` parameter-injection
  mechanism statically excludes a resolver-filled parameter from the
  tool's generated JSON schema (confirmed against the installed
  `mcp==2.1.1` SDK, not assumed), so a client-supplied `tenant_id`/
  `roles`/`auth`-shaped argument has no code path into authorization at
  all, not even a hypothetical validation bug. `rag/mcp/identity.py`'s
  `resolve_http_identity` (per call, from `Context.headers`) and
  `resolve_stdio_identity` (once per process, from `MCP_AUTH_TOKEN`) both
  reuse `api/auth.py`'s `verify_jwt` unmodified, byte-identical rules to
  `api/deps.py:get_current_identity`. `rag/mcp/server.py` deliberately
  does **not** start with `from __future__ import annotations`: the
  SDK resolves each tool parameter's annotation via
  `inspect.signature(fn, eval_str=True)`, which fails with `NameError`
  for a closure-local resolver name under postponed evaluation
  (confirmed by reproducing the crash).
- **Unknown tool arguments are rejected loudly, not silently dropped.**
  The SDK's own dynamically-built per-tool argument model inherits
  Pydantic's default `extra="ignore"` with no exposed strictness knob on
  the high-level `MCPServer.tool()` API (confirmed empirically: an
  injected `tenant_id` field used to be silently dropped, call still
  succeeded). `rag/mcp/server.py`'s `_harden_argument_schemas` closes
  this after registration by reaching into `MCPServer`'s internal
  `ToolManager` and switching each tool's argument model to
  `extra="forbid"` (verified that mutating `model_config` and calling
  `model_rebuild(force=True)` on an already-built Pydantic model does
  take effect), so any unknown field now fails argument validation
  before the tool function or any resolver ever runs, surfacing as a
  normal `CallToolResult(is_error=True)`.
- **The bare mount path works with no client-visible redirect.**
  `Starlette.mount(path, app)`'s `Mount` route only matches `<path>/...`
  (its regex requires the literal `/` delimiter, confirmed against the
  installed `starlette==1.3.1`), so a request to the bare `<path>` (no
  trailing slash) used to fall through to the parent router's own
  `redirect_slashes` handling and 307-redirect to `<path>/`, which the
  MCP SDK's own Streamable HTTP client does not follow during session
  initialization (a real finding from a live Docker container, not a
  unit-test artifact). `rag/mcp/asgi.py`'s `mount_mcp_app` (used by
  `rag/api/main.py` instead of a bare `app.mount()`) layers a small ASGI
  middleware, `_BareMountPathMiddleware`, that rewrites the ASGI scope's
  path server-side only when it exactly equals `mount_path`, so both
  `/mcp` and `/mcp/` work identically, and every other route is
  untouched.
- **Every result is sanitized through the same choke point the agent
  uses.** `rag/mcp/server.py`'s `_run_tool` mirrors
  `agent/graph.py::_execute_tool`'s dispatch-then-sanitize pattern
  exactly, including resolving `auth` a second time
  (`pipeline.resolve_auth`) before the `pipeline.sanitize_evidence` call, the same fix already applied to the in-process agent tools (see
  "Agent-tool fixes" above). No tool exposes raw `VectorStore` results
  directly. `rag/mcp/schemas.py`'s `McpChunkResult` mirrors `SourceItem`'s
  exposure level (never `source_dict()`'s richer internal shape).
- **Dependency.** `mcp` is a core dependency (`pyproject.toml`), not an
  optional extra: it's mounted inside the always-running `rag-api`
  process behind a runtime flag, the same shape as `slowapi`/`pyjwt`, not
  the shape of the eval-only extras (`ragas`/`mlflow`/`cohere`/
  `anthropic`) deliberately excluded from the container image. Adds no
  measurable image-size regression (1.68GB, unchanged from baseline).
- **`security.rate_limit`'s `Limiter` does not wrap the MCP mount**,
  not a regression (MCP requests were never rate-limited before this
  milestone existed either), but a known gap before treating MCP as
  production-hardened to the same degree as `/query`.
- **Stage 1B: `get_customer_case`/`get_case_status`.** `rag/mcp/business/
  store.py` is a small, in-memory, read-only synthetic customer-support-
  case dataset (5 seeded cases across `tenant_alpha`/`tenant_beta`, no
  Postgres/network dependency at all, the point is the integration
  shape, not a real case-management system). Its own tenant/role
  authorization deliberately does not reuse `AuthorizationContext` (a
  case has no document/freshness/trust concept): own tenant plus a
  matching `allowed_roles` entry, or a `security.authorization.
  cross_tenant_support_roles` role also listed on that case, same
  predicate shape and the same config allow-list document ACL already
  uses, not a second parallel privilege list. Unlike document ACL,
  enforcement here is unconditional (no kill-switch): every synthetic
  case has a concrete tenant from creation, so there's no pre-governance
  "`tenant_id IS NULL` means everyone" legacy state to preserve. An
  unauthorized case and a nonexistent one both resolve to `None`,
  mirroring the document corpus's own "indistinguishable from outside"
  property, but because this check runs in Python rather than as an
  opaque SQL filter, a real denial *is* observable: `_lookup_authorized`
  logs a genuine `authorization_denied` audit event (reusing that
  existing `AuthEventType`, not a new one) even though the caller-visible
  response never reveals the distinction. The two tools route through a
  new `_run_business_tool` (structurally parallel to `_run_tool`'s
  tracing/metrics, but skips `pipeline.sanitize_evidence` entirely,
  that method redacts chunk/field-level RAG content, a concept a
  business case doesn't have) and resolve identity via a new
  `_resolve_identity`/`_resolve_identity_only` (factored out of
  `_resolve_auth`, since these tools need a bare `VerifiedIdentity`, not
  an `AuthorizationContext`).
- **Stage 2: the agent as an MCP client, for `get_customer_case`/
  `get_case_status` only.** `src/rag/agent/mcp_client.py` is the only new
  module; `rag/agent/graph.py`'s local `_dispatch_tool` switch (the four
  RAG tools) is untouched. `_execute_tool` gains one branch: a static
  `rag.agent.tool_schemas.REMOTE_MCP_TOOL_NAMES` lookup decides whether a
  decision's `tool_name` routes to the existing local dispatch or to
  `mcp_client.dispatch_remote_tool_sync`, the LLM never reasons about
  "local vs remote," it just picks a tool by name, exactly as before.
  `config.mcp.client.enabled` (default `false`) is a coarse kill-switch
  matching every other milestone's convention: when off, a different,
  4-tool-only prompt (`agent_tool_select_v3.yaml`, unchanged) is loaded
  instead of the new 6-tool `agent_tool_select_v4_mcp.yaml`
  (`config.agent.tool_select_mcp_prompt_path`), and `_execute_tool`
  additionally fails a remote-tool decision closed as a recorded
  `ToolCallRecord` failure (`error="mcp_client_disabled"`) even if one
  somehow arrives anyway, defense in depth, not just prompt omission.
- **Fail-closed without an authenticated identity, no exception.**
  `config.mcp.client.enabled=True` requires `security.auth.enabled=True`, checked once, at process startup
  (`mcp_client.validate_startup_config`, called from `rag/api/main.py`,
  raising `RuntimeError` before the app ever serves a request), not
  deferred to a per-call check. The Stage 1B business tools' tenant/role
  authorization has no kill-switch (see above), so there is no
  trustworthy identity to attach to an agent-originated call when auth is
  off; unlike document-level ACL, there was never a legacy "unrestricted
  by default" state to preserve here. `dispatch_remote_tool_sync` adds a
  second, per-call layer of the same guarantee: a `None`
  `AuthorizationContext`, or one with no `tenant_id`, raises
  `ToolExecutionError` before any network/ASGI call is attempted.
- **Identity propagates as a freshly minted internal service token, never
  the caller's original JWT.** `mcp_client.mint_internal_token` signs a
  short-lived (`internal_token_ttl_seconds`, default 60s) HS256 JWT from
  `AgentState.authorization_context`'s already-*verified*
  `tenant_id`/`roles`, `sub` is always the fixed
  `internal_token_subject` (`"rag-agent-internal"`), never a real
  caller's identity, so an audit-log reader can tell an agent-minted
  service token from a real end-user token at a glance. Only `HS256` is
  supported: it is the only algorithm this codebase holds a *signing*
  (not just verifying) key for (`AppConfig.jwt_signing_key`), so
  `validate_startup_config` also refuses to start with
  `security.auth.jwt.algorithm` set to `RS256`/`ES256`. `iss`/`aud` are
  handled asymmetrically, confirmed directly against the installed
  `pyjwt`, not assumed by analogy: `iss` is always included (a token
  carrying an `iss` claim that the receiving `verify_jwt` isn't
  configured to check is simply never inspected), but `aud` is *omitted
  entirely* when `security.auth.jwt.audience` is unset, since PyJWT
  raises `InvalidAudienceError` for **any** `aud` claim on a token when
  the verifier passes no expected audience to check against, regardless
  of the claim's value, a real failure hit and fixed mid-implementation
  (see `KNOWN_ISSUES.md`), not a hypothetical. When `security.auth.jwt.issuer`/
  `audience` *are* configured, the mint uses those exact values, so the
  unmodified, shared `verify_jwt` genuinely enforces them, never a
  special-cased relaxation for this token.
- **Transport: in-process ASGI by default, real HTTP as an escape
  hatch.** `config.mcp.client.transport="asgi"` (the default) builds an
  `httpx2.ASGITransport` bound directly to the *same* MCP server object
  `rag/api/main.py` mounts (via the new `rag/api/deps.py:
  get_mcp_asgi_app()` singleton, which `main.py` now calls instead of
  constructing its own, a real correctness requirement, not a style
  choice: a second, independently-built `MCPServer` would have a
  session-manager lifespan this process's own lifespan context manager
  never entered). No real socket, no port to configure, still the full
  Streamable HTTP/JSON-RPC protocol end to end (argument hardening,
  identity resolution via headers, structured results). One real bug
  found operating this: the SDK auto-enables DNS-rebinding Host-header
  protection whenever a server is built with the default `host=
  "127.0.0.1"`, allowing only `127.0.0.1:*`/`localhost:*`/`[::1]:*`
  (confirmed directly against `mcp`'s `Server.streamable_http_app`
  source), an arbitrary internal hostname like `"mcp-internal"` is
  rejected with `421 Misdirected Request` before ever reaching a tool.
  `transport="http"` (a real `mcp.client.server_url_env_var`-resolved
  URL, mirroring `OLLAMA_BASE_URL`'s pattern) is the escape hatch for a
  genuinely separate future deployment; `validate_startup_config`
  requires `mcp.enabled=True` only for the `asgi` mode, since `http` mode
  needs no local server object to bind to.
- **Session lifecycle: one fresh `ClientSession` per remote tool call,
  never pooled or reused across calls or requests.** `run_agent()`'s node
  functions are synchronous by design; each remote dispatch bridges to
  the async MCP client via `anyio.run(...)` (the same pattern
  `rag.mcp.stdio_entrypoint.main()` already uses at its own top level),
  safe because both real callers of `run_agent()` (the sync `/agent/
  query` route, and the SSE stream's `run_in_threadpool`-wrapped call)
  already execute off the event-loop thread. Chosen over a persistent
  per-run or cross-request pooled session: trivial identity isolation (no
  risk of one caller's session/token leaking to another), no real
  connection-establishment cost to amortize with the default ASGI
  transport, and remote calls are rare (bounded by the shared
  `max_tool_calls`) against a run that already costs tens of seconds in
  LLM latency, matching this project's repeated "no infrastructure
  without a demonstrated requirement now" pattern.
- **`mcp_remote` evidence is synthetic and structurally excluded from
  document-specific behaviors.** A `SearchResult` gained a fourth
  `origin` value, `"mcp_remote"` (`rag/schemas.py`); `mcp_client.
  _business_result_to_search_result` builds a fabricated `chunk_id`
  (`f"mcp:{tool_name}:{case_id}"`) and `source`
  (`f"mcp://business/{case_id}"`), and deliberately leaves every
  document-governance `ChunkMetadata` field
  (`document_version`/`status`/`effective_from`/`supersedes_source`/
  `allowed_roles`/`classification`/`trust_level`) unset, this evidence
  never passes through `VectorStore`, `RetrievalPipeline.retrieve()`, or
  `expand_with_relationships()`, so freshness resolution, relationship
  expansion, and document-level ACL structurally never apply to it, not
  by a special-case guard but because nothing in its construction path
  calls any of those methods. It still passes through the same
  `pipeline.sanitize_evidence` call every tool's output does (a genuine
  value-add: `detect_injection` runs on a case's free-text `description`
  field), which naturally no-ops on field redaction (no
  `sensitive_field_ids` tag exists for synthetic evidence).
  `tenant_id` on the synthetic chunk is provenance/display only, never
  re-checked as ACL, authorization already happened server-side, inside
  `rag.mcp.business.store`, before this object is ever constructed.
- **No `EvidenceItem` refactor.** Reusing the existing chunk-shaped
  `SearchResult`/`ChunkMetadata` for business-case evidence (rather than
  a new, parallel non-chunk evidence type) was a deliberate, approved
  scope decision for this milestone, it means `_summarize_evidence`,
  `build_context`, and citation rendering in `_synthesize` needed zero
  changes, at the cost of some semantically-overloaded field names
  (`document_id` holding a `case_id`, etc.), documented directly on
  `_business_result_to_search_result`.
- **Tests.** `tests/unit/test_mcp_identity.py` (identity resolution +
  audit-log hygiene, no MCP protocol needed),
  `tests/unit/test_mcp_disabled_is_a_true_noop.py` (`mcp.enabled=false`
  against the real `rag.api.main.app`), `tests/unit/
  test_mcp_business_case_store.py` (plain-function unit tests against
  `rag.mcp.business.store`'s authorization rule, including
  `authorization_denied` audit-log emission via `caplog`),
  `tests/integration/test_mcp_end_to_end.py` (a real MCP client, the
  SDK's own `streamable_http_client`, against a real uvicorn server,
  fake pipeline/vectorstore/embedder doubles, no Postgres/Ollama needed;
  covers all six tools, including a mounted-app test proving both the
  bare-path fix and lifespan propagation together), `tests/integration/
  test_mcp_tenant_isolation.py` (real Postgres, self-skips without it,
  one adversarial scenario per RAG tool, mirroring `test_agent_tool_
  tenant_isolation.py`'s pattern for the underlying, unmodified
  `rag.agent.tools.*` functions, the business tools need no equivalent
  file, since they have no Postgres-backed logic to spot-check).
  Stage 2 adds `tests/unit/test_mcp_client_stage2.py` (18 tests: startup
  validation, token minting including the `iss`/`aud` asymmetry and
  expiry enforcement, fail-closed-without-identity, and the synthetic-
  evidence field-level isolation proof), `tests/unit/
  test_agent_graph_mcp_stage2.py` (7 tests: fail-closed dispatch,
  `max_tool_calls` bounding a mix of local/remote calls, arg-schema
  hardening, `get_related_context` given a synthetic `mcp:...` chunk_id
  staying a safe miss), and `tests/integration/
  test_agent_mcp_client_stage2.py` (5 tests, real wire protocol via the
  ASGI transport against the real business store, no Postgres/Ollama
  needed: an authorized run synthesizing a cited answer, wrong-role and
  cross-tenant denial through the full `run_agent()` path, an MCP-layer
  failure not escaping `run_agent()`, and a captured-header proof that
  the internal token reaching the MCP server is never the original
  caller's credential).

### Stage 3: `update_case_status`, the first write action

`update_case_status(case_id, new_status)` is the one mutating tool in
this codebase, deliberately narrow in scope (one transition table, one
sensitive transition) rather than a general write-action framework.

- **Deterministic, server-side business rules.** `rag.mcp.business.
  store`'s `_VALID_TRANSITIONS` (`open -> in_progress -> resolved ->
  closed`, `closed` terminal) and `_SENSITIVE_TRANSITIONS`
  (`resolved -> closed` only) are never consulted or overridable by the
  LLM. `UpdateCaseStatusArgs` exposes only `case_id`/`new_status`,
  `extra="forbid"`, no approval field of any kind. Already-in-target-
  status is a deterministic no-op (`outcome="already_in_status"`),
  checked before the transition table so it never triggers an approval
  requirement; an invalid transition or a missing approval both leave the
  case untouched, returned as `outcome="invalid_transition"`/
  `"approval_required"`. `identity is None` is a hard denial here
  (unlike `get_customer_case`/`get_case_status`'s "unrestricted"
  convention), a mutating action has no legacy unauthenticated-access
  state worth preserving. The read-check-mutate sequence runs under one
  `_CASE_MUTATION_LOCK`, the first concurrency-safety mechanism this
  store needed, since it's also the first mutator.
- **Approval is trusted-runtime state, never LLM-controlled JSON.** A
  human/ops caller of `POST /agent/query` may include a `case_approvals`
  field naming exact `(case_id, new_status)` pairs, honored only when
  their verified identity holds an `mcp.business_actions.approval_roles`
  role (`rag.api.request_auth.resolve_case_approvals`; a caller lacking
  it has the field silently dropped, logged as `authorization_denied`
  with `action="case_status_approval"`, never trusted). The accepted
  list is stamped onto `AgentState.case_approvals` once, by the router,
  at the same trust tier as `authorization_context`/`filters`. Bounded at
  three points so the internal JWT can't grow unboundedly: a hard
  Pydantic schema ceiling (`rag.mcp.business.schemas.MAX_CASE_APPROVALS`,
  10), a smaller operator-configurable ceiling enforced at the router
  (`mcp.business_actions.max_case_approvals_per_request`, default 5,
  matching `security.dos_limits`'s "runtime config, not a class-level
  constant" convention), and a defensive re-check inside
  `mint_internal_token` itself.
- **The internal token carries the approval, reusing the existing trust
  channel.** `mcp_client.mint_internal_token` gained an optional
  `case_approvals` parameter, embedded as a `case_approvals` claim on the
  same short-lived internal service token Stage 2 already mints, no
  second trust channel invented. `rag.mcp.business.approvals.
  resolve_case_action_approvals` is the MCP-server-side resolver, a new
  module kept independent of `rag.mcp.identity`/`VerifiedIdentity` so the
  general identity model never carries this business-specific concept.
- **A verifiable JWT alone was not sufficient to grant an approval, a
  real authorization bypass found by post-implementation review, not the
  original design or test suite.** The first implementation of
  `resolve_case_action_approvals` verified only that a token was signed
  correctly under `security.auth.jwt`, never that it was actually
  produced by `mint_internal_token`. Since this deployment mints and
  verifies its own tokens in the same trust domain
  (`scripts/issue_dev_token.py` uses the identical shared secret with no
  restriction on which claims it sets), a caller able to obtain any
  ordinary, validly-signed token could attach a `case_approvals` claim to
  it directly and call the MCP server's `update_case_status` tool,
  bypassing `resolve_case_approvals`'s role gate entirely, that gate
  only runs on the `POST /agent/query*` HTTP routes, never for a caller
  that talks to the mounted MCP server directly (an explicitly supported
  access pattern per Stage 1A/1B's own design). Fixed by requiring the
  token's `sub`/`token_use` claims to unambiguously match
  `mint_internal_token`'s fixed markers (`internal_token_subject`,
  `"mcp_internal_service"`), plus an independent re-check of that same
  token's own `roles` claim (the original caller's already-role-gated
  roles, embedded by `mint_internal_token`) against `approval_roles`,
  defense in depth, not a re-statement of the API-boundary check. See
  `KNOWN_ISSUES.md` for the full diagnosis; `tests/unit/
  test_mcp_business_approvals.py` is the dedicated adversarial test file
  this gap's absence of coverage motivated.
- **A write-action attempt is always the last tool call of a run.**
  `rag/agent/graph.py`'s bounded tool-call loop checks `decision.
  tool_name in WRITE_ACTION_TOOL_NAMES` (a `frozenset[str]` in
  `tool_schemas.py`, currently `{"update_case_status"}`) immediately
  after `tool_execute` and, if true, breaks out of the loop regardless of
  outcome (executed, approval_required, invalid_transition,
  already_in_status, or a dispatch failure), proceeding straight to
  synthesis rather than looping back through `evaluate_evidence` to retry
  the same mutation. This was an explicit approval condition (`approval_
  required` must be a terminal outcome for that action attempt, never
  silently retried), implemented as one boolean check rather than a new
  `termination_reason` value or a restructured loop.
- **Config.** `config.mcp.business_actions` (`BusinessActionConfig`):
  `enabled` (default `false`, true no-op, the tool is not registered on
  the MCP server at all, not merely unreachable), `approval_roles`
  (default `["case_status_approver"]`), `max_case_approvals_per_request`
  (default `5`). `config.agent.tool_select_mcp_actions_prompt_path`
  points at a third `select_tool` prompt template
  (`agent_tool_select_v5_mcp_actions.yaml`), loaded only when both
  `mcp.client.enabled` and `mcp.business_actions.enabled` are `true`;
  `v4_mcp`/`v3` stay byte-identical for every other combination.
- **Tests.** `tests/unit/test_mcp_business_case_actions.py` (14 tests,
  plain-function against `rag.mcp.business.store.update_case_status`
  directly, with an autouse fixture snapshotting/restoring the shared
  `_SYNTHETIC_CASES` dict): the full authorization/transition/approval/
  audit matrix. `tests/unit/test_mcp_business_approvals.py` (9 tests, the
  post-review adversarial suite: a genuine internal-service token is
  honored; an ordinary end-user token carrying the identical claim, even
  holding the approval role, is rejected on `sub`; a right-`sub`/
  no-`token_use` token is rejected; an internal-shaped token whose
  embedded `roles` lack an approval role is rejected). Extended
  `test_mcp_client_stage2.py`, `test_agent_graph_mcp_stage2.py`, and
  `test_agent_mcp_client_stage2.py` (real wire protocol: `approval_
  required` with no mutation, a matching approval executing the mutation
  against the live store, a remote failure not crashing `run_agent()`).
