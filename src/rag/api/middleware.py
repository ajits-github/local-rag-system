from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from rag.logging_config import reset_request_id, set_request_id

logger = logging.getLogger("rag.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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
