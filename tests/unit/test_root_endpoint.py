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


def test_root_features_reflect_default_config():
    """`GET /`'s `features` block matches config/default.yaml's own toggles, safely."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    features = response.json()["features"]
    assert features == {
        "auth_enabled": False,
        "insecure_dev_mode": False,
        "authorization_enabled": False,
        "field_redaction_enabled": False,
        "rate_limit_enabled": False,
        "agent_enabled": True,
        "vision_provider": "none",
        "tracing_enabled": False,
    }


def test_root_features_reflect_overridden_config():
    """`GET /`'s `features` block changes when the injected config's security toggles do."""
    config = load_config()
    security = config.security.model_copy(
        update={
            "auth": config.security.auth.model_copy(update={"enabled": True}),
            "field_redaction": config.security.field_redaction.model_copy(update={"enabled": True}),
        }
    )
    patched = config.model_copy(update={"security": security})

    app.dependency_overrides[get_config] = lambda: patched
    try:
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        features = response.json()["features"]
        assert features["auth_enabled"] is True
        assert features["field_redaction_enabled"] is True
        # Untouched toggles are unaffected by the partial override.
        assert features["authorization_enabled"] is False
    finally:
        app.dependency_overrides.pop(get_config, None)


def test_root_features_never_leak_secrets_or_identifying_config():
    """The `features` block never includes a model name, host, key, or connection string.

    A regression guard for the whole point of this endpoint: it must stay
    safe to expose without authentication. Checks the raw JSON text, not
    just the known field names, so a future added field is caught too.
    """
    client = TestClient(app)
    response = client.get("/")
    raw = response.text.lower()

    for forbidden in (
        "qwen",
        "gpt-",
        "claude-",
        "sqlite",
        "postgresql://",
        "api_key",
        "secret",
        "moondream",
    ):
        assert forbidden not in raw, f"GET / leaked {forbidden!r}"


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
