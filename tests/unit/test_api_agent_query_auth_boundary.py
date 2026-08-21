"""Mirrors test_api_query_auth_boundary.py for POST /agent/query.

The same JWT-precedence, forged-claim, and DoS-limit machinery must hold
identically, since agent_query.py reuses rag.api.request_auth rather than
reimplementing it.
"""

from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.deps import (
    get_config,
    get_embedder,
    get_llm,
    get_retrieval_pipeline,
    get_vectorstore,
)
from rag.api.main import app
from rag.config import load_config

_SECRET = "unit-test-only-not-a-real-secret-value"


class _RecordingPipeline:
    """RetrievalPipeline double recording every `answer()` call's `auth` argument.

    Every test in this file ends up on the classic_rag fast path (a
    single `pipeline.answer()` call) -- either because `_StubLLM`
    classifies as 'simple', or because a test explicitly disables the
    agent -- the same behavior these tests exercise for `/query`.
    """

    def __init__(self) -> None:
        """Start with no recorded answer() calls."""
        self.calls: list[dict[str, Any]] = []

    def answer(self, query: str, filters=None, candidate_k=None, auth=None) -> dict[str, Any]:
        """Record the call's `auth` argument and return a fixed classic-RAG answer."""
        self.calls.append({"query": query, "filters": filters, "auth": auth})
        return {
            "answer": "stub answer",
            "sources": [],
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }


def _auth_config(**overrides):
    """Return `load_config()` with JWT auth enabled and `overrides` applied to `security.auth`."""
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    auth_config = auth_config.model_copy(update=overrides)
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _token(**claim_overrides):
    """Build a signed HS256 JWT with default alice/tenant_alpha claims, overridable per test."""
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


class _StubVectorStore:
    """Minimal VectorStore double that only answers health checks."""

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class _StubEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts):
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]


class _StubLLM:
    """LLM double that always classifies as 'simple'.

    `config.agent.enabled` is `True` in the shipped default, so
    `classify_query` genuinely runs (one LLM call) even on a request that
    ends up taking the classic_rag route -- this double reports 'simple'
    so every test here lands on classic_rag regardless of whether the
    agent happens to be enabled or disabled for a given test.
    """

    def generate(self, system: str, user: str) -> str:
        """Classify as simple, so the run always routes to classic_rag."""
        return '{"query_type": "simple"}'

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


@pytest.fixture
def client_with(monkeypatch):
    """Build a TestClient with every agent-endpoint dependency overridden with light doubles."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)

    def _build(config):
        pipeline = _RecordingPipeline()
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_retrieval_pipeline] = lambda: pipeline
        app.dependency_overrides[get_vectorstore] = lambda: _StubVectorStore()
        app.dependency_overrides[get_embedder] = lambda: _StubEmbedder()
        app.dependency_overrides[get_llm] = lambda: _StubLLM()
        return TestClient(app), pipeline

    yield _build
    for dep in (get_config, get_retrieval_pipeline, get_vectorstore, get_embedder, get_llm):
        app.dependency_overrides.pop(dep, None)


def test_forged_body_tenant_id_is_ignored_when_jwt_present(client_with):
    """A body-supplied tenant_id/roles is ignored in favor of the verified JWT's claims."""
    client, pipeline = client_with(_auth_config())
    token = _token(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])

    response = client.post(
        "/agent/query",
        json={"query": "hello", "tenant_id": "tenant_beta", "roles": ["security_admin"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    auth = pipeline.calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["tenant_alpha_operator"]
    assert response.json()["route"] == "classic_rag"


def test_invalid_jwt_never_falls_back_to_unrestricted_retrieval(client_with):
    """A JWT signed with the wrong secret is rejected outright, never treated as unauthenticated."""
    client, pipeline = client_with(_auth_config())
    bad_token = jwt.encode(
        {"sub": "eve", "tenant_id": "tenant_alpha", "roles": ["tenant_alpha_admin"]},
        "wrong-secret-entirely-not-matching",
        algorithm="HS256",
    )

    response = client.post(
        "/agent/query", json={"query": "hello"}, headers={"Authorization": f"Bearer {bad_token}"}
    )

    assert response.status_code == 401
    assert pipeline.calls == []


def test_missing_authorization_header_is_rejected_when_auth_enabled(client_with):
    """With auth enabled and dev mode off, a missing Authorization header is a 401."""
    client, pipeline = client_with(_auth_config(insecure_dev_mode=False))

    response = client.post("/agent/query", json={"query": "hello"})

    assert response.status_code == 401
    assert pipeline.calls == []


def test_auth_disabled_by_default_preserves_body_trusted_behavior(client_with):
    """With auth disabled (the default), body-supplied tenant_id/roles are trusted as-is."""
    config = load_config()
    assert config.security.auth.enabled is False
    client, pipeline = client_with(config)

    response = client.post(
        "/agent/query", json={"query": "hello", "tenant_id": "tenant_alpha", "roles": ["operator"]}
    )

    assert response.status_code == 200
    auth = pipeline.calls[0]["auth"]
    assert auth.tenant_id == "tenant_alpha"
    assert auth.roles == ["operator"]


def test_agent_disabled_response_shape_matches_classic_route(client_with):
    """config.agent.enabled=False: response route is 'classic_rag', no tool calls, no LLM call.

    `config/default.yaml`'s own default is now `True` (the Agentic RAG
    milestone), so this test forces it off explicitly to exercise the
    kill-switch path -- distinct from every other test in this file,
    which lets `_StubLLM` classify its way to `classic_rag` regardless.
    """
    config = load_config()
    agent = config.agent.model_copy(update={"enabled": False})
    config = config.model_copy(update={"agent": agent})
    assert config.agent.enabled is False
    client, _pipeline = client_with(config)

    response = client.post("/agent/query", json={"query": "hello"})

    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "classic_rag"
    assert body["tool_calls"] == []
    assert body["termination_reason"] == "synthesized"


def test_query_endpoint_response_schema_is_unaffected_by_the_new_route(client_with):
    """POST /query itself is untouched -- same 5-field response shape as before."""
    config = load_config()
    client, pipeline = client_with(config)

    response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 200
    assert set(response.json()) == {
        "answer",
        "sources",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    }
