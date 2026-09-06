"""FastAPI application entrypoint: wires up config, middleware, and routers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rag.agent.mcp_client import validate_startup_config as _validate_mcp_client_startup_config
from rag.api.deps import get_config, get_mcp_asgi_app, get_rate_limiter
from rag.api.middleware import RequestIDMiddleware
from rag.api.routers import agent_query, agent_stream, feedback, health, ingest, metrics, query
from rag.audit import log_audit_event
from rag.config import AppConfig
from rag.logging_config import configure_logging
from rag.mcp.asgi import mount_mcp_app
from rag.observability.tracing import configure_tracing

_config = get_config()
configure_logging(_config.app.log_level)
configure_tracing(_config)
_validate_mcp_client_startup_config(_config)

# Built before the FastAPI app itself: when MCP is enabled, the app's own
# lifespan (below) must also enter this sub-app's lifespan, since Starlette
# does not auto-propagate a mounted sub-app's lifespan to its parent, so the
# MCP session manager's background task group would otherwise never start.
# get_mcp_asgi_app() is the single source of truth for this object: the
# agent's own MCP client (rag.agent.mcp_client, transport="asgi") binds
# directly to this same singleton rather than building a second, independent
# MCPServer whose lifespan this process would never enter.
_mcp_app = get_mcp_asgi_app()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Enter the mounted MCP sub-app's lifespan, when MCP is enabled; a no-op otherwise."""
    async with AsyncExitStack() as stack:
        if _mcp_app is not None:
            await stack.enter_async_context(_mcp_app.router.lifespan_context(_mcp_app))
        yield


app = FastAPI(
    title=_config.app.name,
    description="Modular, config-driven local RAG system.",
    lifespan=_lifespan,
)
app.add_middleware(RequestIDMiddleware)

if _mcp_app is not None:
    mount_mcp_app(app, _mcp_app, _config.mcp.server.mount_path)


class FeatureFlags(BaseModel):
    """Safe, non-secret summary of which optional security/feature toggles are active.

    Booleans and provider names only: never a model name, host, connection
    string, JWT setting, or anything else an attacker could use. Lets a
    caller (e.g. the web UI) show the actually-enforced security posture
    rather than assume it from config it cannot see.

    Attributes
    ----------
    auth_enabled : bool
        `security.auth.enabled`; whether a verified JWT is required.
    insecure_dev_mode : bool
        `security.auth.insecure_dev_mode`; only meaningful when
        `auth_enabled` is `True`.
    authorization_enabled : bool
        `security.authorization.enabled`; retrieval-time tenant/role ACL.
    field_redaction_enabled : bool
        `security.field_redaction.enabled`; query-time sensitive-field
        redaction.
    rate_limit_enabled : bool
        `security.rate_limit.enabled`.
    agent_enabled : bool
        `agent.enabled`; whether `POST /agent/query` can route to the
        agent graph, or always takes the `classic_rag` path.
    vision_provider : str
        `vision.provider` (`"none"` or `"ollama"`).
    tracing_enabled : bool
        `observability.tracing.enabled`.
    """

    auth_enabled: bool
    insecure_dev_mode: bool
    authorization_enabled: bool
    field_redaction_enabled: bool
    rate_limit_enabled: bool
    agent_enabled: bool
    vision_provider: str
    tracing_enabled: bool


class RootResponse(BaseModel):
    """Response body for `GET /`."""

    service: str
    status: str
    docs: str
    health: str
    metrics: str | None
    features: FeatureFlags


@app.get("/", response_model=RootResponse)
def root(config: AppConfig = Depends(get_config)) -> RootResponse:
    """Lightweight service/navigation info. No dependency checks. See `GET /health` for that.

    Parameters
    ----------
    config : AppConfig
        Application configuration, read per-request (not the module-level
        `_config` this file also holds) so a `dependency_overrides`-based
        test can exercise the `metrics` link's on/off behavior directly.

    Returns
    -------
    RootResponse
        Service name, status, links to `/health`/`/docs`/`/metrics` (the
        last only when `observability.metrics.enabled`), and a safe
        `FeatureFlags` summary of which optional features are active.
    """
    return RootResponse(
        service=config.app.name,
        status="ok",
        docs="/docs",
        health="/health",
        metrics="/metrics" if config.observability.metrics.enabled else None,
        features=FeatureFlags(
            auth_enabled=config.security.auth.enabled,
            insecure_dev_mode=config.security.auth.insecure_dev_mode,
            authorization_enabled=config.security.authorization.enabled,
            field_redaction_enabled=config.security.field_redaction.enabled,
            rate_limit_enabled=config.security.rate_limit.enabled,
            agent_enabled=config.agent.enabled,
            vision_provider=config.vision.provider,
            tracing_enabled=config.observability.tracing.enabled,
        ),
    )


class RuntimeInfo(BaseModel):
    """Safe, non-secret snapshot of the effective runtime pipeline configuration.

    Complements `FeatureFlags`: that model summarizes security/agent
    toggles for the always-visible feature-flags strip, while this model
    additionally surfaces the concrete provider/model choices behind the
    running pipeline (which model answered, whether BM25/reranking/MCP
    are active), for the developer/debug UI. Kept on a separate `GET
    /info` endpoint rather than folded into `GET /`: `GET /`'s own
    regression test (`test_root_features_never_leak_secrets_or_identifying_config`)
    deliberately asserts no model name ever appears there, and a model
    name is not itself a secret but is exactly the kind of "identifying
    configuration" that endpoint promises never to carry.

    Still bound by the same rule as `FeatureFlags`: never a JWT/signing
    key, DB URL, filesystem path, internal token, raw auth context,
    prompt text, or tenant data. Provider names and model identifiers are
    the only "new" kind of information here relative to `FeatureFlags`.

    Attributes
    ----------
    generation_provider : str
        `generation.provider` (e.g. `"ollama"`).
    generation_model : str
        `generation.model_name`.
    embedding_provider : str
        `embedding.provider`.
    embedding_model : str
        `embedding.model_name`.
    vectorstore_provider : str
        `vectorstore.provider` (e.g. `"pgvector"`).
    retrieval_mode : str
        `retrieval.provider` (`"dense"` or `"hybrid"`).
    sparse_retrieval_enabled : bool
        Whether BM25 keyword search runs alongside dense search
        (`retrieval.provider == "hybrid"`).
    fusion_method : str | None
        `"rrf"` when `sparse_retrieval_enabled`, else `None` (no fusion
        step runs in dense-only mode).
    reranker_provider : str
        `reranker.provider`.
    reranker_enabled : bool
        `reranker.provider != "none"`.
    agent_enabled : bool
        `agent.enabled`.
    mcp_enabled : bool
        `mcp.enabled`; whether the MCP server is mounted at all.
    mcp_client_enabled : bool
        `mcp.client.enabled`; whether the agent dispatches the two
        business-case tools as a real MCP client call.
    vision_provider : str
        `vision.provider` (`"none"` or `"ollama"`).
    tracing_enabled : bool
        `observability.tracing.enabled`.
    rate_limit_enabled : bool
        `security.rate_limit.enabled`.
    field_redaction_enabled : bool
        `security.field_redaction.enabled`.
    authorization_enabled : bool
        `security.authorization.enabled`.
    auth_enabled : bool
        `security.auth.enabled`.
    """

    generation_provider: str
    generation_model: str
    embedding_provider: str
    embedding_model: str
    vectorstore_provider: str
    retrieval_mode: str
    sparse_retrieval_enabled: bool
    fusion_method: str | None
    reranker_provider: str
    reranker_enabled: bool
    agent_enabled: bool
    mcp_enabled: bool
    mcp_client_enabled: bool
    vision_provider: str
    tracing_enabled: bool
    rate_limit_enabled: bool
    field_redaction_enabled: bool
    authorization_enabled: bool
    auth_enabled: bool


@app.get("/info", response_model=RuntimeInfo)
def info(config: AppConfig = Depends(get_config)) -> RuntimeInfo:
    """Effective runtime pipeline configuration, safe to expose without authentication.

    Lightweight, config-only (no dependency injection of vectorstore/llm/
    pipeline), matching `root()`'s own pattern. Intended for a developer/
    debug UI panel, not for making authorization decisions.

    Parameters
    ----------
    config : AppConfig
        Application configuration, read per-request so a
        `dependency_overrides`-based test can exercise this directly.

    Returns
    -------
    RuntimeInfo
        The safe, non-secret runtime configuration snapshot.
    """
    sparse_retrieval_enabled = config.retrieval.provider == "hybrid"
    return RuntimeInfo(
        generation_provider=config.generation.provider,
        generation_model=config.generation.model_name,
        embedding_provider=config.embedding.provider,
        embedding_model=config.embedding.model_name,
        vectorstore_provider=config.vectorstore.provider,
        retrieval_mode=config.retrieval.provider,
        sparse_retrieval_enabled=sparse_retrieval_enabled,
        fusion_method="rrf" if sparse_retrieval_enabled else None,
        reranker_provider=config.reranker.provider,
        reranker_enabled=config.reranker.provider != "none",
        agent_enabled=config.agent.enabled,
        mcp_enabled=config.mcp.enabled,
        mcp_client_enabled=config.mcp.client.enabled,
        vision_provider=config.vision.provider,
        tracing_enabled=config.observability.tracing.enabled,
        rate_limit_enabled=config.security.rate_limit.enabled,
        field_redaction_enabled=config.security.field_redaction.enabled,
        authorization_enabled=config.security.authorization.enabled,
        auth_enabled=config.security.auth.enabled,
    )


app.state.limiter = get_rate_limiter()


def _handle_rate_limit_exceeded(request: Request, exc: Exception) -> JSONResponse:
    """Return a 429 for a rate-limited request and emit an audit event.

    Parameters
    ----------
    request : Request
        The rate-limited HTTP request.
    exc : Exception
        The `RateLimitExceeded` raised by `slowapi`. Typed as the base
        `Exception` to match `Starlette.add_exception_handler`'s expected
        handler signature.

    Returns
    -------
    JSONResponse
        A 429 response with the exceeded-limit detail.
    """
    detail = getattr(exc, "detail", str(exc))
    log_audit_event("rate_limit_exceeded", path=request.url.path)
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {detail}"})


app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(agent_query.router)
app.include_router(agent_stream.router)
app.include_router(feedback.router)
app.include_router(metrics.router)
