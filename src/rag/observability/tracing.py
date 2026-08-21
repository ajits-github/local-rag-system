"""OpenTelemetry tracing setup: configurable, defensive, OTLP-exporting.

Disabled by default (`config.observability.tracing.enabled=False`): the
OpenTelemetry API's own no-op `TracerProvider` stays in place, so every
`start_span` call anywhere in this codebase is free. No "is tracing
enabled" branching needed at any instrumentation call site.
`configure_tracing` is the only place that ever imports
`opentelemetry.sdk`/the OTLP exporter; every other module only depends on
the always-installed `opentelemetry.api`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import Span

if TYPE_CHECKING:
    from rag.config import AppConfig

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer("rag")
_configured = False


def configure_tracing(config: AppConfig) -> None:
    """Install a real OTLP-exporting `TracerProvider`, if tracing is enabled.

    Idempotent: a second call is a no-op once tracing has already been
    configured. Any exporter/SDK construction failure is caught and
    logged rather than raised, so a misconfigured OTLP endpoint can never
    prevent the API from starting.

    Parameters
    ----------
    config : AppConfig
        Application configuration; `config.observability.tracing` and
        `config.otlp_endpoint()` supply every setting.
    """
    global _tracer, _configured
    if _configured or not config.observability.tracing.enabled:
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        tracing_cfg = config.observability.tracing
        provider = TracerProvider(
            resource=Resource.create({"service.name": tracing_cfg.service_name}),
            sampler=TraceIdRatioBased(tracing_cfg.sample_ratio),
        )
        exporter = OTLPSpanExporter(endpoint=f"{config.otlp_endpoint()}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("rag")
        _configured = True
    except Exception:
        logger.warning(
            "Failed to configure OpenTelemetry tracing; continuing without it", exc_info=True
        )


def reset_tracing_for_tests() -> None:
    """Reset the module-level configured flag and tracer. Test-only helper."""
    global _tracer, _configured
    _tracer = trace.get_tracer("rag")
    _configured = False


@contextmanager
def start_span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
    """Open a span, defensively.

    Safe to call unconditionally at any instrumentation point: when
    tracing is disabled this is the OpenTelemetry API's own no-op span
    (near-zero cost), and any unexpected error from span creation,
    attribute-setting, or teardown is caught and logged rather than
    propagated. A broken exporter/SDK can never turn into a 500 on
    `/query` or `/agent/query`. An exception raised by the *caller's* code
    inside the `with` block is never swallowed here; it propagates
    normally after the span is closed (and recorded as an error on the
    span, when tracing is actually active).

    Parameters
    ----------
    name : str
        The span name.
    attributes : Mapping[str, Any] | None, optional
        Initial span attributes; `None` values are dropped (OpenTelemetry
        rejects them) rather than raising.

    Yields
    ------
    Span
        The active span (or `trace.INVALID_SPAN` if span creation itself
        failed).
    """
    span_cm = None
    span: Span = trace.INVALID_SPAN
    try:
        span_cm = _tracer.start_as_current_span(name)
        span = span_cm.__enter__()
        if attributes:
            set_attributes(span, attributes)
    except Exception:
        logger.warning("Failed to start telemetry span %r", name, exc_info=True)
        span_cm = None

    try:
        yield span
    except BaseException:
        exc_info = sys.exc_info()
        if span_cm is not None:
            try:
                span_cm.__exit__(*exc_info)
            except Exception:
                logger.warning("Failed to close telemetry span %r", name, exc_info=True)
        raise
    else:
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Failed to close telemetry span %r", name, exc_info=True)


def set_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    """Set span attributes, dropping `None` values; never raises.

    Parameters
    ----------
    span : Span
        The span to annotate.
    attributes : Mapping[str, Any]
        Attribute key/value pairs. Never pass raw retrieved content,
        credentials, or model reasoning text here. See
        `docs/architecture.md`'s "Observability" section for the full
        never-attach list.
    """
    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        logger.warning("Failed to set telemetry span attributes", exc_info=True)
