# MCP

--8<-- "README.md:docs-mcp"

Every existing security control this codebase already built -- JWT
identity, tenant/role authorization, freshness, trust filtering,
field-level redaction, prompt-injection detection, audit logging -- stays
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

None of these reimplement retrieval or authorization logic -- they are
thin MCP adapters over the exact same, already-adversarially-tested
functions `rag/agent/graph.py`'s in-process tool dispatch calls.

## Two synthetic business-backend tools (Stage 1B)

| Tool | What it does | Underlying function |
|---|---|---|
| `get_customer_case` | Fetch a synthetic customer-support case's full detail by `case_id` | `rag.mcp.business.store.get_customer_case` |
| `get_case_status` | Fetch only a case's status, priority, and last-updated time -- a narrower read | `rag.mcp.business.store.get_case_status` |

These exist to demonstrate MCP as an integration layer to a separate
backend/business system, not just another transport for this
deployment's own RAG tools. `rag.mcp.business.store` is a small,
in-memory, read-only synthetic case dataset (no Postgres/network
dependency, a stand-in for a real backend a production MCP server might
front) with its own tenant/role authorization: a case's owning tenant
plus a matching role on the case, or a
`security.authorization.cross_tenant_support_roles` role that is also
listed on that case (the same rule, and the same config list, document
retrieval already uses for cross-tenant access -- not a second parallel
privilege list). A case the caller may not access is returned as `null`,
identical to a case that doesn't exist -- unlike document-level ACL
(a SQL predicate with no visibility into what it excluded), this
in-Python check *can* observe a real denial, so it logs an
`authorization_denied` audit event even though the response itself never
reveals whether the case exists.

## Identity: transport-resolved, never a tool argument

A verified JWT's identity is resolved once per call from the
Streamable HTTP request's own headers (`resolve_http_identity`), or once
per process from the `MCP_AUTH_TOKEN` environment variable for the
stdio transport (`resolve_stdio_identity`) -- both reuse the same
`verify_jwt` the HTTP `/query` boundary already uses. The MCP SDK's
`Resolve()` parameter-injection mechanism statically excludes the
resolved `auth` value from every tool's generated JSON schema, so a
client cannot supply or override `tenant_id`/`roles`/`auth` even in
principle, not just by convention. Governed entirely by the existing
`security.auth` config tree -- there is no separate, weaker `mcp.auth.*`
toggle that could drift from it.

## Two hardening fixes worth knowing about

- **Unknown tool arguments are rejected loudly**, not silently dropped.
  The MCP SDK's own generated schema doesn't set
  `additionalProperties: false` by default; this codebase hardens every
  tool's argument model to `extra="forbid"` after registration, so an
  injected or arbitrary unknown field fails validation before the tool
  function ever runs.
- **The bare mount path (`/mcp`, no trailing slash) works reliably.**
  Starlette's `Mount` route only matches `<mount_path>/...`, so the bare
  path used to 307-redirect to the trailing-slash form -- a redirect the
  MCP SDK's own client doesn't follow during session init. A small
  server-side ASGI middleware now makes both spellings work identically.

## Stage 2: the agent as an MCP client

Stages 1A/1B are server-only: this codebase exposes tools, it doesn't
call any as a client. Stage 2 closes that other half, but only for the
two business tools -- `rag/agent/mcp_client.py`, dispatched from
`rag/agent/graph.py`'s bounded tool loop. The four RAG tools stay exactly
what they already were: direct, in-process function calls, never routed
through MCP.

- **Fails closed, not open.** `mcp.client.enabled=True` requires
  `security.auth.enabled=True` -- checked once at process startup, not
  per request -- since the business tools' authorization has no
  kill-switch to fall back on.
- **Never forwards the caller's own token.** Each remote call mints a
  fresh, short-lived internal service token from the caller's already-
  verified tenant/roles (`sub` is always a fixed, clearly synthetic
  marker), signed with the same secret the receiving, unmodified
  `verify_jwt` already checks against.
- **In-process by default, real network as an escape hatch.** The
  default transport binds an ASGI transport directly to the same MCP
  server object this process mounts -- no socket, still the full
  Streamable HTTP protocol end to end. Pointing it at a genuinely
  separate deployment later is a config change, not new code.
- **One session per call**, never pooled across requests -- simplest
  correct answer to "does this call need an already-running event loop,"
  since `run_agent()`'s node functions are synchronous by design.
- **Business-case evidence is synthetic and inert to document logic.**
  A fabricated `chunk_id`/`source`, `origin="mcp_remote"`, and every
  freshness/relationship/ACL-relevant field left unset -- this evidence
  never touches `VectorStore` or the retrieval pipeline, so those
  subsystems structurally never see it.

Full design writeup -- token-minting's `iss`/`aud` asymmetry (a real
PyJWT behavior found mid-implementation), the DNS-rebinding Host-header
bug the ASGI transport hit, and the session-lifecycle tradeoffs:
[MCP Integration](../architecture.md#mcp-integration).

## Config

```yaml
mcp:
  enabled: false      # true no-op when false: the MCP server is never built or mounted
  server:
    mount_path: /mcp
  client:
    enabled: false    # true no-op when false: the agent never offers or dispatches the two remote tools
    transport: asgi   # or "http", for a genuinely separate future deployment
```

`mcp.enabled: false` and `mcp.client.enabled: false` are both shipped
defaults. When `mcp.enabled` is `false`, the MCP server -- and its
Streamable HTTP session manager -- is never even constructed, not merely
built and left unmounted; `GET /mcp` 404s exactly like a route that was
never registered. When `mcp.client.enabled` is `false`, the agent's
tool-select prompt never offers `get_customer_case`/`get_case_status`,
and a decision naming either one anyway fails closed as an ordinary
recorded tool failure, never a real dispatch attempt.

Full design writeup, including both server-side hardening fixes' root
causes, the alternatives considered and rejected for the bare-mount-path
fix, and the full Stage 2 design:
[MCP Integration](../architecture.md#mcp-integration).

API reference: [MCP](../reference/mcp.md).
