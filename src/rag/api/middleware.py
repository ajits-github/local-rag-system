"""ASGI middleware that stamps every request with a correlation id."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.logging_config import reset_request_id, set_request_id

logger = logging.getLogger("rag.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Sets the request-id contextvar for a request's lifetime and logs its duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Set the request id, run the handler, log duration, and echo the id back.

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
        try:
            response = await call_next(request)
        finally:
            logger.info(
                "request_handled",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            reset_request_id(token)
        response.headers["x-request-id"] = request_id
        return response
