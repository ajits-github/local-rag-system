"""Prometheus metrics: metric names, bounded labels, and the /metrics endpoint's on/off behavior.

Follows `test_api_dos_limits.py`'s TestClient + `dependency_overrides`
pattern for the endpoint-level assertions.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.deps import get_config
from rag.api.main import app
from rag.config import load_config
from rag.observability import metrics


def test_render_metrics_includes_expected_metric_families():
    """Every documented metric family name appears in the rendered exposition text."""
    metrics.observe_http_request("GET", "/health", 200, 0.01)
    metrics.observe_agent_request("agent", 1.2, 4)
    metrics.observe_tool_call("search_knowledge_base", True, 0.2)
    metrics.observe_node_latency("classify", 0.1, 0.09)
    metrics.observe_retrieval_latency("dense", 0.05)
    metrics.observe_termination_reason("synthesized")
    metrics.observe_evidence_sufficiency(True)
    metrics.observe_error("tool")

    text = metrics.render_metrics().decode("utf-8")

    for family in [
        "rag_http_requests_total",
        "rag_http_request_duration_seconds",
        "rag_agent_requests_total",
        "rag_agent_total_latency_seconds",
        "rag_agent_steps",
        "rag_agent_tool_calls_total",
        "rag_agent_tool_latency_seconds",
        "rag_agent_node_latency_seconds",
        "rag_agent_node_llm_latency_seconds",
        "rag_retrieval_latency_seconds",
        "rag_agent_termination_reason_total",
        "rag_agent_evidence_sufficiency_total",
        "rag_errors_total",
    ]:
        assert family in text, f"missing metric family: {family}"


def test_no_high_cardinality_labels_in_exported_text():
    """Never a query string, tenant/document/chunk id, or arbitrary tool argument as a label."""
    metrics.observe_tool_call("search_knowledge_base", True, 0.1)
    text = metrics.render_metrics().decode("utf-8")

    forbidden_substrings = [
        "tenant_id=",
        "document_id=",
        "chunk_id=",
        "query=",
        "user_id=",
    ]
    for forbidden in forbidden_substrings:
        assert forbidden not in text


def _client_with_config(**metrics_overrides) -> TestClient:
    config = load_config()
    metrics_cfg = config.observability.metrics.model_copy(update=metrics_overrides)
    observability_cfg = config.observability.model_copy(update={"metrics": metrics_cfg})
    patched = config.model_copy(update={"observability": observability_cfg})
    app.dependency_overrides[get_config] = lambda: patched
    return TestClient(app)


def test_metrics_endpoint_returns_200_when_enabled():
    """`GET /metrics` returns Prometheus text exposition when `observability.metrics.enabled`."""
    client = _client_with_config(enabled=True)
    try:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
    finally:
        app.dependency_overrides.pop(get_config, None)


def test_metrics_endpoint_returns_404_when_disabled():
    """`GET /metrics` returns 404 when `observability.metrics.enabled` is `False`."""
    client = _client_with_config(enabled=False)
    try:
        response = client.get("/metrics")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_config, None)
