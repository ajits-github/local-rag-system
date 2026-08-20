"""`GET /`: lightweight service info, no dependency injection of vectorstore/llm/pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.deps import get_config
from rag.api.main import app
from rag.config import load_config


def test_root_returns_service_info_and_metrics_link_when_enabled():
    """`GET /` reports status and links, including `/metrics` when it's enabled."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert body["metrics"] == "/metrics"
    assert "service" in body


def test_root_omits_metrics_link_when_metrics_disabled():
    """`GET /` reports `metrics: null` when `observability.metrics.enabled` is `False`."""
    config = load_config()
    metrics_cfg = config.observability.metrics.model_copy(update={"enabled": False})
    observability = config.observability.model_copy(update={"metrics": metrics_cfg})
    patched = config.model_copy(update={"observability": observability})

    app.dependency_overrides[get_config] = lambda: patched
    try:
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["metrics"] is None
    finally:
        app.dependency_overrides.pop(get_config, None)
