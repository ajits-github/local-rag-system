"""ASGI middleware that stamps every request with a correlation id."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.logging_config import reset_request_id, set_request_id
from rag.observability import metrics as observability_metrics
from rag.observability import tracing

logger = logging.getLogger("rag.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Sets the request-id contextvar for a request's lifetime and records its telemetry.

    Also the one place HTTP-level observability lives: opens a root
    OpenTelemetry span per request (so agent-graph spans opened deeper in
    the call stack nest under it) and records
    `rag_http_requests_total`/`rag_http_request_duration_seconds`. This
    method already computes `duration_ms` and has `method`/route in
    scope, so it's the natural single place for both, rather than a
    second middleware duplicating the same bookkeeping.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Set the request id, run the handler, record telemetry, and echo the id back.

        Parameters
        ----------
        request : Request
            The incoming request.
        call_next : RequestResponseEndpoint
            The next ASGI handler in the middleware chain.

        Returns
        -------
        Response
            The handler's response, with an `x-request-id` header set.
        """
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = set_request_id(request_id)
        start = time.monotonic()
        response: Response | None = None
        span_cm = tracing.start_span(f"{request.method} {request.url.path}")
        span = span_cm.__enter__()
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            status_code = response.status_code if response is not None else 500
            route = request.scope.get("route")
            path_label = getattr(route, "path", None) or request.url.path
            tracing.set_attributes(
                span,
                {
                    "http.method": request.method,
                    "http.route": path_label,
                    "http.status_code": status_code,
                    "request_id": request_id,
                },
            )
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Failed to close HTTP request span", exc_info=True)
            observability_metrics.observe_http_request(
                request.method, path_label, status_code, duration_ms / 1000
            )
            logger.info(
                "request_handled",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            reset_request_id(token)
        response.headers["x-request-id"] = request_id
        return response
