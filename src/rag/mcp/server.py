"""Builds the MCP server exposing the four core RAG tools plus two synthetic business tools.

Every RAG tool here is a thin adapter over `rag.agent.tools.*` -- the
exact same functions the in-process agent graph calls (see
`rag.agent.graph._dispatch_tool`). No retrieval or authorization logic is
reimplemented. `auth`/`identity` is resolved once per call from the
transport (see `rag.mcp.identity`) via the MCP SDK's `Resolve`
parameter-injection mechanism, which statically excludes resolver-filled
parameters from a tool's generated JSON schema -- confirmed directly
against the installed SDK (see `tests/integration/test_mcp_end_to_end.py`),
not assumed -- so no tool call can ever supply or override it. Every RAG
result is sanitized through `RetrievalPipeline.sanitize_evidence` at one
central dispatch helper (`_run_tool`) before it is serialized and
returned, mirroring `rag.agent.graph._execute_tool`'s dispatch-then-
sanitize pattern exactly, including resolving `auth` a second time for
the sanitize call itself (the same fix documented for the in-process
agent tools).

`get_customer_case`/`get_case_status` (Stage 1B) are a second, structurally
independent tool family: thin adapters over `rag.mcp.business.store`, a
synthetic customer-support-case backend with its own tenant/role
authorization (see that module's docstring for why it does not reuse
`AuthorizationContext`/`sanitize_evidence`). They exist to demonstrate MCP
as an integration layer to a separate backend/business system, not just
another transport for this deployment's own RAG tools -- so they
deliberately do not route through `_dispatch`/`_run_tool` at all.

Verified against the installed `mcp==2.1.1` SDK (v2's `MCPServer`,
formerly `FastMCP` in v1 -- a hard rename, not a deprecation warning).

Every registered tool's argument model is hardened after registration
(see `_harden_argument_schemas`) so an unknown field -- including an
attempted `tenant_id`/`roles`/`auth` injection -- is rejected loudly by
Pydantic rather than silently dropped: the SDK's own dynamically-built
per-tool argument model inherits Pydantic's default `extra="ignore"`
with no exposed strictness knob on the high-level `MCPServer.tool()`
decorator (confirmed directly against the installed SDK source). This
was already structurally harmless (no tool parameter binds those names,
and `auth` is exclusively resolver-injected -- see `_resolve_auth`), but
a silent drop is a worse failure mode than a loud rejection at a
security boundary.

Deliberately does NOT start with ``from __future__ import annotations``,
unlike every other module in this codebase: the SDK resolves each tool's
parameter annotations via `inspect.signature(fn, eval_str=True)`, which
`eval()`s a postponed (string) annotation against the function's
`__globals__` -- and `_resolve_auth` below is a closure-local name, never
present in module globals. Under postponed evaluation this fails with a
`NameError` at server-build time (confirmed directly: this was hit while
writing this module, not a hypothetical). Keeping annotations as real,
already-evaluated objects here is required, not a style regression.
"""

import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, Literal, TypeVar

from mcp.server.mcpserver import Context, MCPServer, Resolve
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from rag.agent import tools
from rag.agent.tool_schemas import (
    ContentTypeFilter,
    GetDocumentArgs,
    GetLatestDocumentArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
)
from rag.api.auth import VerifiedIdentity
from rag.api.request_auth import build_authorization_context
from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.mcp.business import store as business_store
from rag.mcp.business.schemas import CaseStatusResult, CustomerCase
from rag.mcp.identity import resolve_http_identity
from rag.mcp.schemas import McpChunkResult, to_mcp_result
from rag.observability import metrics as observability_metrics
from rag.observability import tracing
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

_ToolName = Literal[
    "search_knowledge_base", "get_document", "get_latest_document", "get_related_context"
]
_BusinessToolName = Literal["get_customer_case", "get_case_status"]
_T = TypeVar("_T")


class _Unset:
    """Sentinel type distinguishing "not passed" from an explicit `None` for `fixed_identity`.

    A dedicated class, not a bare `object()`, so mypy can narrow
    `fixed_identity is _UNSET` correctly: `object` would collapse the
    `VerifiedIdentity | None | object` union down to `object` and defeat
    narrowing entirely.
    """


_UNSET = _Unset()


def _harden_argument_schemas(server: MCPServer) -> None:
    """Make every registered tool's argument model reject an unknown field, loudly.

    Reaches into `MCPServer`'s internal `ToolManager` (no higher-level
    hook exists in this SDK version -- confirmed by inspection, not
    assumed; closing this fully would mean dropping to the SDK's
    lower-level `Server` API, out of scope for this fix) to switch each
    tool's dynamically-built Pydantic argument model from the default
    `extra="ignore"` to `extra="forbid"`, then regenerates the tool's
    already-cached JSON schema (what `tools/list` advertises to a
    client) so it correctly advertises `additionalProperties: false`
    too. Verified empirically: mutating `model_config` and calling
    `model_rebuild(force=True)` on an already-constructed Pydantic model
    does take effect on the next `model_validate` call.

    An unknown field -- including an attempted `tenant_id`/`roles`/
    `auth` injection -- now fails argument validation before the tool
    function or any resolver ever runs (`Tool.run` calls
    `fn_metadata.validate_arguments` first), surfacing to the caller as
    a normal `CallToolResult(is_error=True)` (a `pydantic.ValidationError`
    is caught and re-raised as `ToolError` by the SDK itself). This never
    changes what `auth` resolves to: `auth` is excluded from every tool's
    argument model entirely (see `Resolve(_resolve_auth)` below), so it
    was never reachable through argument validation in the first place --
    this only closes the *unknown-key* gap.
    """
    for tool in server._tool_manager.list_tools():
        arg_model = tool.fn_metadata.arg_model
        arg_model.model_config["extra"] = "forbid"
        arg_model.model_rebuild(force=True)
        tool.parameters = arg_model.model_json_schema(by_alias=True)


def build_mcp_server(
    config: AppConfig,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    embedder: Embedder,
    *,
    fixed_identity: VerifiedIdentity | None | _Unset = _UNSET,
) -> MCPServer:
    """Build the MCP server exposing the four RAG tools and two business-case tools.

    Parameters
    ----------
    config : AppConfig
        Application configuration; `security.auth` governs identity
        resolution (see `rag.mcp.identity`), `agent.max_tool_top_k`/
        `agent.max_chunks_per_document_fetch*` govern the same
        server-controlled bounds the in-process agent tools already use.
    pipeline : RetrievalPipeline
        The process-wide retrieval pipeline singleton.
    vectorstore : VectorStore
        The process-wide vector store singleton.
    embedder : Embedder
        The process-wide embedder singleton, used by `get_document`/
        `get_latest_document`'s relevance-selection pass.
    fixed_identity : VerifiedIdentity | None, keyword-only
        Left unset (the default) for the Streamable-HTTP transport:
        identity is resolved per tool call from that call's own request
        headers (see `rag.mcp.identity.resolve_http_identity`). Pass an
        explicit value (including `None`, meaning "auth disabled/
        unrestricted") only for the stdio transport, which has no
        per-request headers at all and instead resolves one identity
        once at process startup (see
        `rag.mcp.identity.resolve_stdio_identity` and
        `rag.mcp.stdio_entrypoint`) and reuses it for every call in that
        process. This parameter is the only difference between the two
        transports' identity handling -- everything else in this
        function is shared, so neither transport can silently drift from
        the other.

    Returns
    -------
    MCPServer
        A server with all six tools registered. Callers mount it via
        `.streamable_http_app()` (see `rag.mcp.asgi`) or run it directly
        via `.run_stdio_async()` (see `rag.mcp.stdio_entrypoint`).
    """
    server: MCPServer = MCPServer(
        name="local-rag-system",
        instructions=(
            "Search and fetch content from this deployment's authorized knowledge "
            "base, and look up synthetic customer-support cases from a separate "
            "backend system. Every result is already scoped to your verified "
            "identity -- you cannot and should not supply a tenant, role, or "
            "authorization value yourself; any such argument is ignored."
        ),
    )

    # Resolved once, here in the outer scope: mypy does not preserve `is`/`isinstance`
    # narrowing of a variable captured by a nested closure (confirmed directly --
    # narrowing fixed_identity inside _resolve_auth itself left its type as
    # `VerifiedIdentity | _Unset | None`), so the _Unset case is collapsed into a
    # stable, already-narrowed `VerifiedIdentity | None` local before any closure
    # reads it.
    use_fixed_identity = not isinstance(fixed_identity, _Unset)
    resolved_fixed_identity: VerifiedIdentity | None = (
        fixed_identity if isinstance(fixed_identity, VerifiedIdentity) else None
    )

    def _resolve_identity(ctx: Context) -> VerifiedIdentity | None:
        """Resolve this call's verified identity from the transport, never from tool args.

        The shared first half of `_resolve_auth` below, factored out so
        the business-case tools (which need a bare `VerifiedIdentity`,
        not an `AuthorizationContext`) can reuse the exact same
        transport-resolution rule rather than a second copy of it.
        """
        return (
            resolved_fixed_identity
            if use_fixed_identity
            else resolve_http_identity(ctx.headers, config)
        )

    async def _resolve_auth(
        ctx: Context, as_of: date | None, require_trust_level: str | None
    ) -> AuthorizationContext | None:
        """Build this call's `AuthorizationContext` from the transport, never from tool args.

        `as_of`/`require_trust_level` are ordinary, non-privileged tool
        arguments (matching `AgentQueryRequest`'s own fields) -- not
        identity claims -- so they are read by name from the calling
        tool, same as `build_authorization_context` already treats them
        for `/query`/`/agent/query`. `tenant_id`/`roles` are always
        passed as `None`: this project's transports never source them
        from anything but a verified identity.
        """
        identity = _resolve_identity(ctx)
        return build_authorization_context(identity, None, None, as_of, require_trust_level)

    async def _resolve_identity_only(ctx: Context) -> VerifiedIdentity | None:
        """`Resolve()`-compatible async wrapper around `_resolve_identity`.

        The business-case tools declare `identity: Annotated[VerifiedIdentity
        | None, Resolve(_resolve_identity_only)]` -- they need the bare
        identity, not an `AuthorizationContext`, so they resolve through
        this wrapper rather than `_resolve_auth`. `Resolve()` requires an
        async callable (matching `_resolve_auth`'s own signature,
        confirmed against the installed SDK), which is the only reason
        this thin wrapper exists separately from `_resolve_identity`
        itself.
        """
        return _resolve_identity(ctx)

    def _dispatch(
        tool_name: _ToolName,
        args: Any,
        *,
        auth: AuthorizationContext | None,
        filters: dict[str, Any] | None,
        dataset_id: str | None,
        query: str,
    ) -> list[SearchResult]:
        """Call the matching `rag.agent.tools` function.

        The same dispatch `agent.graph._dispatch_tool` makes.
        """
        if tool_name == "search_knowledge_base":
            clamped_top_k = min(args.top_k, config.agent.max_tool_top_k)
            clamped_args = args.model_copy(update={"top_k": clamped_top_k})
            return list(tools.search_knowledge_base(clamped_args, pipeline, filters, auth))
        if tool_name == "get_document":
            chunks = tools.get_document(
                args,
                pipeline,
                vectorstore,
                dataset_id,
                query,
                embedder,
                auth,
                config.agent.max_chunks_per_document_fetch,
                config.agent.max_chunks_per_document_fetch_hard_ceiling,
            )
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        if tool_name == "get_latest_document":
            chunks = tools.get_latest_document(
                args,
                pipeline,
                vectorstore,
                dataset_id,
                query,
                embedder,
                auth,
                config.agent.max_chunks_per_document_fetch,
                config.agent.max_chunks_per_document_fetch_hard_ceiling,
            )
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        if tool_name == "get_related_context":
            chunks = tools.get_related_context(args, pipeline, vectorstore, auth, dataset_id)
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in chunks]
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )  # unreachable: tool_name is Literal-validated

    def _run_tool(
        tool_name: _ToolName,
        args: Any,
        *,
        auth: AuthorizationContext | None,
        filters: dict[str, Any] | None,
        dataset_id: str | None,
        query: str,
    ) -> list[McpChunkResult]:
        """Dispatch, sanitize (the single universal choke point), observe, and serialize.

        Every one of the four tool handlers below routes through this
        one function -- mirroring `agent.graph._execute_tool`'s design
        note that sanitization is applied centrally, "not delegated
        per-tool -- so no tool can bypass it by construction."
        """
        t0 = time.perf_counter()
        try:
            with tracing.start_span(tool_name, attributes={"tool_name": tool_name}) as span:
                results = _dispatch(
                    tool_name, args, auth=auth, filters=filters, dataset_id=dataset_id, query=query
                )
                tracing.set_attributes(span, {"tool_success": True, "result_count": len(results)})
        except tools.ToolExecutionError as exc:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            raise ToolError(str(exc)) from None
        except Exception:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            logger.exception("MCP tool %s failed", tool_name)
            raise  # an unanticipated failure: let the SDK report it generically, never leak details

        effective_auth = pipeline.resolve_auth(
            auth, {"dataset_id": dataset_id} if dataset_id else None
        )
        sanitized = pipeline.sanitize_evidence(results, effective_auth)
        latency_seconds = time.perf_counter() - t0
        observability_metrics.observe_tool_call(tool_name, True, latency_seconds)
        return [to_mcp_result(r) for r in sanitized]

    def _run_business_tool(tool_name: _BusinessToolName, fn: Callable[[], _T]) -> _T:
        """Dispatch one business-case tool: observe, then delegate authorization to `fn`.

        Structurally parallel to `_run_tool` (tracing span, latency
        metric, `mcp_tool` error counter on an unanticipated exception),
        but deliberately does not call `pipeline.sanitize_evidence` --
        that method redacts chunk/field-level content, a concept this
        resource type doesn't have. `rag.mcp.business.store` already
        returns `None` for both "not found" and "not authorized"; there
        is nothing left for this function to sanitize or gate further.
        `fn` is a zero-arg closure so this stays generic over which
        `store` function (and therefore which return type) it wraps,
        rather than branching on `tool_name` itself.
        """
        t0 = time.perf_counter()
        try:
            with tracing.start_span(tool_name, attributes={"tool_name": tool_name}) as span:
                result = fn()
                tracing.set_attributes(span, {"tool_success": True, "found": result is not None})
        except Exception:
            latency_seconds = time.perf_counter() - t0
            observability_metrics.observe_tool_call(tool_name, False, latency_seconds)
            observability_metrics.observe_error("mcp_tool")
            logger.exception("MCP tool %s failed", tool_name)
            raise  # an unanticipated failure: let the SDK report it generically, never leak details

        latency_seconds = time.perf_counter() - t0
        observability_metrics.observe_tool_call(tool_name, True, latency_seconds)
        return result

    @server.tool(
        name="search_knowledge_base",
        description=(
            "Search the authorized knowledge base for chunks relevant to a query. "
            "Results are already scoped to your verified identity, freshness-"
            "filtered, and redacted."
        ),
    )
    async def search_knowledge_base(
        query: str,
        top_k: Annotated[int, Field(ge=1, le=20)] = 5,
        content_type: ContentTypeFilter | None = None,
        dataset_id: str | None = None,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = SearchKnowledgeBaseArgs(query=query, top_k=top_k, content_type=content_type)
        filters = {"dataset_id": dataset_id} if dataset_id else None
        return _run_tool(
            "search_knowledge_base",
            args,
            auth=auth,
            filters=filters,
            dataset_id=dataset_id,
            query=query,
        )

    @server.tool(
        name="get_document",
        description=(
            "Fetch a specific authorized document by its source path, bounded and "
            "relevance-selected against your current question."
        ),
    )
    async def get_document(
        source: str,
        dataset_id: str,
        query: str,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetDocumentArgs(source=source)
        return _run_tool(
            "get_document", args, auth=auth, filters=None, dataset_id=dataset_id, query=query
        )

    @server.tool(
        name="get_latest_document",
        description=(
            "Resolve a document's source path to its currently-effective version "
            "(if superseded), then fetch it, bounded and relevance-selected "
            "against your current question."
        ),
    )
    async def get_latest_document(
        source: str,
        dataset_id: str,
        query: str,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetLatestDocumentArgs(source=source)
        return _run_tool(
            "get_latest_document", args, auth=auth, filters=None, dataset_id=dataset_id, query=query
        )

    @server.tool(
        name="get_related_context",
        description=(
            "Fetch parent/neighbor context for an already-retrieved chunk, by its "
            "chunk_id (copy the chunk_id value from a prior result verbatim)."
        ),
    )
    async def get_related_context(
        chunk_id: str,
        dataset_id: str | None = None,
        as_of: date | None = None,
        require_trust_level: str | None = None,
        auth: Annotated[AuthorizationContext | None, Resolve(_resolve_auth)] = None,
    ) -> list[McpChunkResult]:
        args = GetRelatedContextArgs(chunk_id=chunk_id)
        return _run_tool(
            "get_related_context", args, auth=auth, filters=None, dataset_id=dataset_id, query=""
        )

    @server.tool(
        name="get_customer_case",
        description=(
            "Fetch a synthetic customer-support case by its case_id, from a separate "
            "backend business system (not this deployment's knowledge base). Scoped "
            "to your verified tenant and role; a case you may not access is "
            "indistinguishable from one that doesn't exist."
        ),
    )
    async def get_customer_case(
        case_id: str,
        identity: Annotated[VerifiedIdentity | None, Resolve(_resolve_identity_only)] = None,
    ) -> CustomerCase | None:
        cross_tenant_support_roles = config.security.authorization.cross_tenant_support_roles
        return _run_business_tool(
            "get_customer_case",
            lambda: business_store.get_customer_case(case_id, identity, cross_tenant_support_roles),
        )

    @server.tool(
        name="get_case_status",
        description=(
            "Fetch only the status, priority, and last-updated time of a synthetic "
            "customer-support case by its case_id -- a narrower read than "
            "get_customer_case for callers that only need to check case state."
        ),
    )
    async def get_case_status(
        case_id: str,
        identity: Annotated[VerifiedIdentity | None, Resolve(_resolve_identity_only)] = None,
    ) -> CaseStatusResult | None:
        cross_tenant_support_roles = config.security.authorization.cross_tenant_support_roles
        return _run_business_tool(
            "get_case_status",
            lambda: business_store.get_case_status(case_id, identity, cross_tenant_support_roles),
        )

    _harden_argument_schemas(server)
    return server
