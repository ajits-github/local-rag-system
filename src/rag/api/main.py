"""FastAPI application entrypoint: wires up config, middleware, and routers."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rag.api.deps import get_config, get_rate_limiter
from rag.api.middleware import RequestIDMiddleware
from rag.api.routers import agent_query, agent_stream, health, ingest, metrics, query
from rag.audit import log_audit_event
from rag.config import AppConfig
from rag.logging_config import configure_logging
from rag.observability.tracing import configure_tracing

_config = get_config()
configure_logging(_config.app.log_level)
configure_tracing(_config)

app = FastAPI(
    title=_config.app.name,
    description="Modular, config-driven local RAG system.",
)
app.add_middleware(RequestIDMiddleware)


@app.get("/")
def root(config: AppConfig = Depends(get_config)) -> dict[str, Any]:
    """Lightweight service/navigation info. No dependency checks -- see `GET /health` for that.

    Parameters
    ----------
    config : AppConfig
        Application configuration, read per-request (not the module-level
        `_config` this file also holds) so a `dependency_overrides`-based
        test can exercise the `metrics` link's on/off behavior directly.

    Returns
    -------
    dict[str, Any]
        Service name, status, and links to `/health`/`/docs`/`/metrics`
        (the last only when `observability.metrics.enabled`).
    """
    return {
        "service": config.app.name,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics" if config.observability.metrics.enabled else None,
    }


app.state.limiter = get_rate_limiter()


def _handle_rate_limit_exceeded(request: Request, exc: Exception) -> JSONResponse:
    """Return a 429 for a rate-limited request and emit an audit event.

    Parameters
    ----------
    request : Request
        The rate-limited HTTP request.
    exc : Exception
        The `RateLimitExceeded` raised by `slowapi` -- typed as the base
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
