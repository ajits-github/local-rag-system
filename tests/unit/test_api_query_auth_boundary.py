from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_current_identity, get_retrieval_pipeline
from rag.api.main import app
from rag.config import load_config

_SECRET = "unit-test-only-not-a-real-secret-value"


class _RecordingPipeline:
    """RetrievalPipeline double recording every `answer()` call's `auth` argument."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        """Record the call and return a minimal QueryResponse-shaped dict."""
        self.calls.append({"query": query, "filters": filters, "auth": auth})
        return {
            "answer": "stub answer",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }


def _auth_config(**overrides):
    """Load config with security.auth enabled and HS256 configured for tests."""
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    auth_config = auth_config.model_copy(update=overrides)
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _token(**claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


@pytest.fixture
def client_with(monkeypatch):
    """Build a TestClient with `get_config`/`get_retrieval_pipeline` overridden.

    Returns a `(client, pipeline)` factory: `client_with(config)` installs
    that config and a fresh `_RecordingPipeline`, cleaning up overrides
    after the test.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    installed: list[Any] = []

    def _build(config):
        pipeline = _RecordingPipeline()
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
        installed.append(config)
        return TestClient(app), pipeline

    yield _build
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_retrieval_pipeline, None)


def test_forged_body_tenant_id_is_ignored_when_jwt_present(client_with):
    """A verified JWT's tenant_id wins over a conflicting request-body tenant_id."""
    client, pipeline = client_with(_auth_config())
    token = _token(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])

    response = client.post(
        "/query",
        json={"query": "hello", "tenant_id": "tenant_beta", "roles": ["security_admin"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert len(pipeline.calls) == 1
    auth = pipeline.calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["tenant_alpha_operator"]


def test_forged_body_roles_is_ignored_when_jwt_present(client_with):
    """A verified JWT's roles win over request-body roles, even a privileged-looking forgery."""
    client, pipeline = client_with(_auth_config())
    token = _token(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])

    response = client.post(
        "/query",
        json={"query": "hello", "roles": ["system_admin", "security_admin"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    auth = pipeline.calls[0]["auth"]
    assert auth.roles == ["tenant_alpha_operator"]
    assert "system_admin" not in auth.roles


def test_invalid_jwt_never_falls_back_to_unrestricted_retrieval(client_with):
    """An invalid-signature token is rejected with 401. the pipeline is never called."""
    client, pipeline = client_with(_auth_config())
    bad_token = jwt.encode(
        {"sub": "eve", "tenant_id": "tenant_alpha", "roles": ["tenant_alpha_admin"]},
        "wrong-secret-entirely-not-matching",
        algorithm="HS256",
    )

    response = client.post(
        "/query", json={"query": "hello"}, headers={"Authorization": f"Bearer {bad_token}"}
    )

    assert response.status_code == 401
    assert pipeline.calls == []


def test_missing_authorization_header_is_rejected_when_auth_enabled(client_with):
    """No Authorization header at all is rejected with 401 when insecure_dev_mode is off."""
    client, pipeline = client_with(_auth_config(insecure_dev_mode=False))

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 401
    assert pipeline.calls == []


def test_insecure_dev_mode_accepts_body_fields_only_when_no_jwt_present(client_with):
    """insecure_dev_mode falls back to body-trusted fields only when no token was sent at all."""
    client, pipeline = client_with(_auth_config(insecure_dev_mode=True))

    response = client.post(
        "/query", json={"query": "hello", "tenant_id": "tenant_alpha", "roles": ["dev_role"]}
    )

    assert response.status_code == 200
    auth = pipeline.calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["dev_role"]


def test_insecure_dev_mode_still_rejects_an_invalid_present_token(client_with):
    """insecure_dev_mode never overrides a present-but-invalid JWT. still fails closed."""
    client, pipeline = client_with(_auth_config(insecure_dev_mode=True))

    response = client.post(
        "/query", json={"query": "hello"}, headers={"Authorization": "Bearer garbage-token"}
    )

    assert response.status_code == 401
    assert pipeline.calls == []


def test_auth_disabled_by_default_preserves_body_trusted_behavior(client_with):
    """With security.auth.enabled=False (the system default), body fields are trusted unchanged."""
    config = load_config()
    assert config.security.auth.enabled is False
    client, pipeline = client_with(config)

    response = client.post(
        "/query", json={"query": "hello", "tenant_id": "tenant_alpha", "roles": ["operator"]}
    )

    assert response.status_code == 200
    auth = pipeline.calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["operator"]


def test_get_current_identity_returns_none_when_auth_disabled():
    """get_current_identity() returns None (not raise) when auth is disabled. unit-level."""
    from fastapi import Request
    from starlette.datastructures import Headers

    config = load_config()
    assert config.security.auth.enabled is False

    scope = {"type": "http", "headers": Headers({}).raw, "method": "POST", "path": "/query"}
    request = Request(scope)

    identity = get_current_identity(request, config)

    assert identity is None
