"""FastAPI application entrypoint: wires up config, middleware, and routers."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import psycopg2.errors
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg2.pool import PoolError
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
from rag.observability import metrics as observability_metrics
from rag.observability.tracing import configure_tracing

logger = logging.getLogger(__name__)


def _detect_worker_count() -> int | None:
    """Best-effort process-worker-count signal from the `WEB_CONCURRENCY` env var.

    `WEB_CONCURRENCY` is a common Gunicorn/Uvicorn convention for the
    number of worker processes a supervisor spawns; nothing guarantees an
    operator sets it, so `None` means "unknown," never "one."

    Returns
    -------
    int | None
        The parsed worker count, or `None` when unset or unparseable.
    """
    raw = os.environ.get("WEB_CONCURRENCY")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _warn_if_multi_worker_mcp_business_actions(config: AppConfig) -> None:
    """Warn (never raise) when >1 worker is likely running with a mutating MCP tool enabled.

    `rag.mcp.business.store`'s in-memory `_SYNTHETIC_CASES` dict and every
    `api/deps.py` `lru_cache` singleton are process-local: with
    `--workers N > 1`, each worker process would silently hold its own,
    independently-mutated copy of the case store, breaking cross-worker
    consistency for `update_case_status` with no error at all. Nothing in
    this deployment currently sets `--workers` above 1, and the
    `WEB_CONCURRENCY` convention isn't universally set/reliable, so this
    only ever logs a warning, never raises -- a false negative (an actual
    multi-worker deployment this check can't see) is possible, but a
    false positive should never block startup.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    """
    if not config.mcp.business_actions.enabled:
        return
    worker_count = _detect_worker_count()
    if worker_count is None or worker_count <= 1:
        return
    logger.warning(
        "mcp_business_actions_multi_worker_risk",
        extra={"worker_count": worker_count},
    )


_config = get_config()
configure_logging(_config.app.log_level)
configure_tracing(_config)
_validate_mcp_client_startup_config(_config)
_warn_if_multi_worker_mcp_business_actions(_config)

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
    toggles, while this one exposes the concrete provider/model choices
    behind the running pipeline (which model answered, whether
    BM25/reranking/MCP are active) for a developer/debug UI. Kept on a
    separate `GET /info` endpoint since `GET /` promises never to reveal
    a model name or other identifying configuration.

    Never exposes a JWT/signing key, DB URL, filesystem path, internal
    token, raw auth context, prompt text, or tenant data.

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


def _handle_pool_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean 503 when the DB connection pool is exhausted, no internal detail leaked.

    `psycopg2.pool.ThreadedConnectionPool.getconn()` raises `PoolError`
    immediately on exhaustion rather than queuing the caller, so a burst
    of concurrent DB-touching requests beyond the pool's `maxconn` would
    otherwise surface as an unhandled 500 with an internal exception
    message in the response body.

    Parameters
    ----------
    request : Request
        The request that hit an exhausted pool.
    exc : Exception
        The `PoolError` raised. Typed as the base `Exception` to match
        `Starlette.add_exception_handler`'s expected handler signature.

    Returns
    -------
    JSONResponse
        A 503 response with a generic detail message.
    """
    log_audit_event("database_pool_exhausted", path=request.url.path)
    observability_metrics.observe_error("database")
    return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable"})


def _handle_integrity_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean 503 for a DB integrity-constraint violation, no internal detail leaked.

    Catches `psycopg2.errors.IntegrityError` (and its subclasses, e.g.
    `UniqueViolation`) as defense-in-depth: whether or not a given write
    path already resolves this at the application level (e.g. an atomic
    upsert), an uncaught constraint violation must never surface as an
    unhandled 500 with a raw database error message in the response body.

    Parameters
    ----------
    request : Request
        The request whose write violated a DB constraint.
    exc : Exception
        The `IntegrityError` raised. Typed as the base `Exception` to
        match `Starlette.add_exception_handler`'s expected handler
        signature.

    Returns
    -------
    JSONResponse
        A 503 response with a generic detail message.
    """
    log_audit_event("database_integrity_violation", path=request.url.path)
    observability_metrics.observe_error("database")
    return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable"})


app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)
app.add_exception_handler(PoolError, _handle_pool_error)
app.add_exception_handler(psycopg2.errors.IntegrityError, _handle_integrity_error)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(agent_query.router)
app.include_router(agent_stream.router)
app.include_router(feedback.router)
app.include_router(metrics.router)
