"""Tracing disabled-by-default no-op behavior, and defensiveness against a broken exporter/span.

`configure_tracing` is never called by these tests, so the module-level
`_tracer` stays the OpenTelemetry API's own default no-op tracer -- the
same state a fresh process is in before `rag.api.main` calls
`configure_tracing` at import time with `tracing.enabled=False`.
"""

from __future__ import annotations

from opentelemetry import trace

from rag.observability import tracing


def test_start_span_never_raises_when_tracing_is_not_configured():
    """A no-op span, including a `None`-valued attribute, is created without raising."""
    with tracing.start_span("some-node", attributes={"a": 1, "b": None}) as span:
        assert span is not None


def test_start_span_yields_a_usable_span_even_when_disabled():
    """Calling methods on a disabled (no-op) span never raises."""
    with tracing.start_span("classify") as span:
        # A no-op span's context is invalid, but calling methods on it must never raise.
        span.set_attribute("extra", "value")
        ctx = span.get_span_context()
        assert ctx is not None


def test_start_span_propagates_caller_exceptions_without_swallowing_them():
    """The caller's own exception inside a `with start_span(...)` block is never swallowed."""

    class Boom(Exception):
        """Marker exception raised inside the span block to prove it propagates."""

    raised = False
    try:
        with tracing.start_span("classify"):
            raise Boom("caller failure")
    except Boom:
        raised = True
    assert raised, "start_span must never swallow the caller's own exception"


def test_start_span_survives_a_broken_tracer(monkeypatch):
    """If span creation itself fails, start_span logs and yields an invalid span, never raises."""

    def _broken_start_as_current_span(*args, **kwargs):
        raise RuntimeError("simulated exporter/tracer failure")

    monkeypatch.setattr(tracing._tracer, "start_as_current_span", _broken_start_as_current_span)

    with tracing.start_span("classify") as span:
        assert span is trace.INVALID_SPAN


def test_set_attributes_drops_none_values_and_never_raises():
    """`set_attributes` silently skips `None`-valued entries rather than raising."""
    with tracing.start_span("classify") as span:
        tracing.set_attributes(span, {"a": None, "b": "value", "c": 1})
