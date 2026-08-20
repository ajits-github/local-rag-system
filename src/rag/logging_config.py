"""Structured JSON logging with request-id and trace-id propagation.

This module owns log formatting only; OpenTelemetry tracing itself lives
in `rag.observability.tracing` (see `docs/architecture.md`'s
"Observability" section for the full split between the two). Every log
line still carries the `request_id` contextvar this module has always
set, and now additionally carries `trace_id`/`span_id` when a real
(non-no-op) span is active, so a log line and the trace it happened
inside can be cross-referenced without a second correlation mechanism.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_RESERVED_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


def set_request_id(value: str | None) -> contextvars.Token:
    """Set the current request id, returning a token to restore the prior value.

    Parameters
    ----------
    value : str | None
        The request id to install for the current context.

    Returns
    -------
    contextvars.Token
        Token to pass to `reset_request_id` when the request completes.
    """
    return _request_id_var.set(value)


def reset_request_id(token: contextvars.Token) -> None:
    """Restore the request id to its value before the matching `set_request_id` call.

    Parameters
    ----------
    token : contextvars.Token
        The token returned by the corresponding `set_request_id` call.
    """
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    """Return the request id for the current context, if any.

    Returns
    -------
    str | None
        The current request id, or ``None`` outside of a request.
    """
    return _request_id_var.get()


def _current_trace_ids() -> tuple[str, str] | None:
    """Return `(trace_id, span_id)` as hex strings, or `None` if no real span is active.

    Deliberately does not import `rag.observability` at module scope --
    this module is imported very early (before config is loaded), and
    this keeps the two modules' import order irrelevant. Never raises:
    a missing/no-op span context is the common case (tracing disabled by
    default), not an error.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:
        return None


class JSONFormatter(logging.Formatter):
    """Renders each `logging.LogRecord` as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize `record` to a JSON string, including the request id and any extras.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format.

        Returns
        -------
        str
            A single JSON-encoded log line.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        trace_ids = _current_trace_ids()
        if trace_ids is not None:
            payload["trace_id"], payload["span_id"] = trace_ids
        extras = {
            key: val
            for key, val in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_ATTRS and not key.startswith("_")
        }
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install a JSON-formatted stdout handler as the root logger's sole handler.

    Parameters
    ----------
    level : str, optional
        Root logger level, by default ``"INFO"``.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
