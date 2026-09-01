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

## Config

```yaml
mcp:
  enabled: false      # true no-op when false: the MCP server is never built or mounted
  server:
    mount_path: /mcp
```

`mcp.enabled: false` is the shipped default. When `false`, the MCP
server -- and its Streamable HTTP session manager -- is never even
constructed, not merely built and left unmounted; `GET /mcp` 404s
exactly like a route that was never registered.

## What's deferred

Stage 1A (the four RAG tools) and Stage 1B (the two synthetic
business-case tools) are both server-only. Making the in-process agent
itself an MCP client (Stage 2) is still deliberately deferred until
there's a concrete need for it -- tracked in the README's Roadmap
section, not started or scaffolded.

Full design writeup, including both hardening fixes' root causes and the
alternatives considered and rejected for the bare-mount-path fix:
[MCP Integration](../architecture.md#mcp-integration).

API reference: [MCP](../reference/mcp.md).
