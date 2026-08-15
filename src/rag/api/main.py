"""FastAPI application entrypoint: wires up config, middleware, and routers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rag.api.deps import get_config, get_rate_limiter
from rag.api.middleware import RequestIDMiddleware
from rag.api.routers import health, ingest, query
from rag.audit import log_audit_event
from rag.logging_config import configure_logging

_config = get_config()
configure_logging(_config.app.log_level)

app = FastAPI(
    title=_config.app.name,
    description="Modular, config-driven local RAG system.",
)
app.add_middleware(RequestIDMiddleware)

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
