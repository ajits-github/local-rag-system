"""Observability config defaults and env-resolved accessor.

Metrics/live-events default on (in-process only, no external dependency,
same "harmless default-on" precedent as MLflowConfig); tracing defaults
off (needs an OTLP endpoint actually listening). see config.py's
`ObservabilityConfig`/`TracingConfig` docstrings for the full reasoning.
"""

from __future__ import annotations

import os

from rag.config import load_config


def test_metrics_and_live_events_default_enabled_tracing_default_disabled():
    """Confirms the shipped `config/default.yaml` defaults, not just the pydantic model defaults."""
    config = load_config()

    assert config.observability.metrics.enabled is True
    assert config.observability.metrics.path == "/metrics"
    assert config.observability.tracing.enabled is False
    assert config.observability.live_events.enabled is True


def test_otlp_endpoint_defaults_to_localhost_when_env_var_unset(monkeypatch):
    """Native/uvicorn-on-host default, mirroring `ollama_base_url()`'s fallback pattern."""
    config = load_config()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert config.otlp_endpoint() == "http://localhost:4318"


def test_otlp_endpoint_reads_current_process_environment(monkeypatch):
    """Resolved as a method (not a frozen field) so it reflects the *current* environment."""
    config = load_config()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")

    assert config.otlp_endpoint() == "http://jaeger:4318"
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://jaeger:4318"
