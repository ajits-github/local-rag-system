from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.datastructures import Headers

from rag.api.auth import VerifiedIdentity
from rag.api.deps import _rate_limit_key, get_rate_limiter


def _scope(path: str = "/query") -> dict[str, Any]:
    """Build a fresh Starlette ASGI scope dict -- never share one across two `Request`s."""
    return {"type": "http", "headers": Headers({}).raw, "method": "POST", "path": path}


def test_limit_is_scoped_per_tenant_not_global():
    """_rate_limit_key buckets by tenant_id when an identity is present on request.state."""
    request_alpha = Request(_scope())
    request_alpha.state.identity = VerifiedIdentity(
        subject="alice", tenant_id="tenant_alpha", roles=[]
    )
    request_beta = Request(_scope())
    request_beta.state.identity = VerifiedIdentity(subject="bob", tenant_id="tenant_beta", roles=[])

    key_alpha = _rate_limit_key(request_alpha)
    key_beta = _rate_limit_key(request_beta)

    assert key_alpha != key_beta
    assert key_alpha == "tenant:tenant_alpha"
    assert key_beta == "tenant:tenant_beta"


def test_key_falls_back_to_client_ip_when_no_identity():
    """With no request.state.identity set (e.g. auth disabled), the key falls back to client IP."""
    request = Request(_scope())
    request.state.identity = None

    key = _rate_limit_key(request)

    assert key.startswith("ip:")


def test_disabled_by_default_limiter_never_blocks():
    """get_rate_limiter() reads the real (rate_limit.enabled=false) default config -- a no-op.

    query.py's `_limiter` (the object its `@_limiter.limit(...)` decorator is
    actually bound to) is fixed at `rag.api.routers.query` import time --
    same "no hot-reload" pattern as every other AppConfig-derived singleton
    in `deps.py`. This asserts that process-start value is correctly
    disabled given the shipped default config; enforcement mechanics
    themselves are proven separately below against an isolated app, since
    the production app's decorator can't be swapped per-test.
    """
    limiter = get_rate_limiter()
    assert limiter.enabled is False


def test_requests_over_threshold_return_429():
    """The slowapi Limiter+decorator+middleware combination enforces a configured 2/minute limit.

    Built against a small, isolated FastAPI app using the exact same
    pattern as `api/main.py`/`api/routers/query.py` (Limiter, `@limiter.
    limit(...)`, `SlowAPIMiddleware`, a `RateLimitExceeded` handler) --
    not `rag.api.main.app` itself, since that app's `/query` route is
    permanently bound to whichever `Limiter` instance existed at `query.py`
    import time (config read once at process start, never swapped later,
    matching this project's established no-hot-reload convention for every
    other config-derived singleton). This test proves the mechanism itself
    enforces correctly; `test_disabled_by_default_limiter_never_blocks`
    confirms the shipped default leaves it off.
    """

    def _handle_rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": "rate limited"})

    limiter = Limiter(key_func=lambda request: "fixed-test-key", enabled=True)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/probe")
    @limiter.limit("2/minute")
    def probe(request: Request) -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)

    first = client.get("/probe")
    second = client.get("/probe")
    third = client.get("/probe")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
