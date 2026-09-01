"""FastAPI application entrypoint: wires up config, middleware, and routers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rag.api.deps import (
    get_config,
    get_embedder,
    get_rate_limiter,
    get_retrieval_pipeline,
    get_vectorstore,
)
from rag.api.middleware import RequestIDMiddleware
from rag.api.routers import agent_query, agent_stream, health, ingest, metrics, query
from rag.audit import log_audit_event
from rag.config import AppConfig
from rag.logging_config import configure_logging
from rag.mcp.asgi import build_mcp_asgi_app
from rag.observability.tracing import configure_tracing

_config = get_config()
configure_logging(_config.app.log_level)
configure_tracing(_config)

# Built before the FastAPI app itself: when MCP is enabled, the app's own
# lifespan (below) must also enter this sub-app's lifespan, since Starlette
# does not auto-propagate a mounted sub-app's lifespan to its parent -- the
# MCP session manager's background task group would otherwise never start.
_mcp_app = (
    build_mcp_asgi_app(_config, get_retrieval_pipeline(), get_vectorstore(), get_embedder())
    if _config.mcp.enabled
    else None
)


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
    app.mount(_config.mcp.server.mount_path, _mcp_app)


class FeatureFlags(BaseModel):
    """Safe, non-secret summary of which optional features are currently active.

    Boolean toggles and provider names only, matching the same shape as
    `config/default.yaml`'s own "swap point" fields. Never a model name,
    host, connection string, JWT setting, or any other value an attacker
    could use; every field here is already either directly observable by
    probing the API's behavior or explicitly documented as safe-to-expose
    in `CLAUDE.md`. Exists so a caller (in particular, this project's own
    web UI) can render an at-a-glance "what's actually enforced right now"
    summary instead of discovering the active security posture by
    accident. See `CLAUDE.md`'s "Web UI" section for the incident that
    prompted this.

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
app.include_router(metrics.router)
