"""Structured `request_handled` log content: status_code, request_id, trace correlation, privacy.

Uses `fastapi.testclient.TestClient` against the real `rag.api.main.app`
and the always-available `GET /` route (no DI overrides needed -- it only
depends on `get_config`, which loads the real default config with no I/O;
see `test_api_middleware.py`'s docstring for why `/health` is avoided
here: it resolves `get_vectorstore`/`get_llm`, which do real I/O).

Asserts against the actual `JSONFormatter`-rendered JSON (not just the raw
`LogRecord`'s `extra` attributes), since `request_id`/`trace_id`/`span_id`
are synthesized by the formatter from the ambient contextvar/span at
format time, not stamped onto the record itself. A dedicated in-memory
handler formats each record synchronously via the real `JSONFormatter`,
the same object `logging_config.configure_logging()` installs, so this
exercises the exact code path `docker logs`/stdout would show -- without
fighting pytest's own stdout capturing, which can't see output written
through a `logging.StreamHandler` bound to `sys.stdout` at import time.
"""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider

from rag.api.main import app
from rag.logging_config import JSONFormatter
from rag.observability import tracing

client = TestClient(app)


class _JSONCapture(logging.Handler):
    """Formats every emitted record via the real `JSONFormatter` and parses it back."""

    def __init__(self) -> None:
        """Install the real `JSONFormatter` and start with an empty capture list."""
        super().__init__()
        self.setFormatter(JSONFormatter())
        self.payloads: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Format `record` and store the parsed JSON payload."""
        self.payloads.append(json.loads(self.format(record)))


def _capture(logger_name: str, make_request) -> list[dict]:
    """Run `make_request()` while capturing every record on `logger_name` as real JSON."""
    logger = logging.getLogger(logger_name)
    handler = _JSONCapture()
    logger.addHandler(handler)
    try:
        make_request()
    finally:
        logger.removeHandler(handler)
    return handler.payloads


def _request_handled(payloads: list[dict]) -> list[dict]:
    """Filter captured payloads down to `request_handled` entries."""
    return [p for p in payloads if p.get("message") == "request_handled"]


def test_request_handled_includes_status_code_and_core_fields():
    """The structured log itself carries status_code, not just the span/metric."""
    payloads = _capture("rag.api", lambda: client.get("/"))

    records = _request_handled(payloads)
    assert len(records) == 1
    payload = records[0]
    assert payload["status_code"] == 200
    assert payload["method"] == "GET"
    assert payload["path"] == "/"
    assert payload["duration_ms"] >= 0.0
    assert payload["route_name"]  # resolved for a real, matched route


def test_request_handled_includes_status_code_for_a_404():
    """A route that never matches still gets its real status_code logged, not a default."""
    response_holder: list[int] = []

    def _make_request() -> None:
        response_holder.append(client.get("/definitely-not-a-real-route").status_code)

    payloads = _capture("rag.api", _make_request)

    assert response_holder == [404]
    payload = _request_handled(payloads)[-1]
    assert payload["status_code"] == 404


def test_request_id_present_even_when_tracing_is_disabled():
    """request_id is always populated; trace_id/span_id are absent when tracing is off."""
    payloads = _capture("rag.api", lambda: client.get("/"))

    payload = _request_handled(payloads)[-1]
    assert payload["request_id"]
    assert "trace_id" not in payload
    assert "span_id" not in payload


def test_request_handled_includes_trace_and_span_id_when_tracing_is_active(monkeypatch):
    """When a real span is active, trace_id/span_id are attached to the same log line.

    Regression guard: `request_handled` used to be logged *after* the HTTP
    root span had already been closed, so the ambient trace-context lookup
    always found nothing even with tracing enabled.
    """
    real_tracer = TracerProvider().get_tracer("test")
    monkeypatch.setattr(tracing, "_tracer", real_tracer)

    payloads = _capture("rag.api", lambda: client.get("/"))

    payload = _request_handled(payloads)[-1]
    assert len(payload["trace_id"]) == 32
    assert len(payload["span_id"]) == 16


def test_request_handled_never_logs_the_authorization_header():
    """An inbound Authorization header never appears anywhere in the rendered JSON line."""
    forged_token = "Bearer eyJhbGciOiJIUzI1NiJ9.super-secret-payload.signature"

    payloads = _capture("rag.api", lambda: client.get("/", headers={"Authorization": forged_token}))

    payload = _request_handled(payloads)[-1]
    payload_text = json.dumps(payload)
    assert forged_token not in payload_text
    assert "super-secret-payload" not in payload_text
