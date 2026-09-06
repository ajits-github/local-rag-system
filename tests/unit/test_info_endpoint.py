"""`GET /info`: safe runtime pipeline configuration, no DI of vectorstore/llm/pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.deps import get_config
from rag.api.main import app
from rag.config import load_config


def test_info_reflects_default_config():
    """`GET /info` reports the effective provider/model choices from config/default.yaml."""
    client = TestClient(app)
    response = client.get("/info")

    assert response.status_code == 200
    body = response.json()
    config = load_config()
    assert body["generation_provider"] == config.generation.provider
    assert body["generation_model"] == config.generation.model_name
    assert body["embedding_provider"] == config.embedding.provider
    assert body["embedding_model"] == config.embedding.model_name
    assert body["vectorstore_provider"] == config.vectorstore.provider
    assert body["retrieval_mode"] == "dense"
    assert body["sparse_retrieval_enabled"] is False
    assert body["fusion_method"] is None
    assert body["reranker_provider"] == "none"
    assert body["reranker_enabled"] is False
    assert body["mcp_enabled"] is False
    assert body["mcp_client_enabled"] is False


def test_info_reflects_hybrid_retrieval_and_reranker_config():
    """Hybrid retrieval reports sparse_retrieval_enabled/fusion_method; a reranker reports on."""
    config = load_config()
    retrieval = config.retrieval.model_copy(update={"provider": "hybrid"})
    reranker = config.reranker.model_copy(update={"provider": "cross_encoder"})
    patched = config.model_copy(update={"retrieval": retrieval, "reranker": reranker})

    app.dependency_overrides[get_config] = lambda: patched
    try:
        client = TestClient(app)
        response = client.get("/info")
        assert response.status_code == 200
        body = response.json()
        assert body["retrieval_mode"] == "hybrid"
        assert body["sparse_retrieval_enabled"] is True
        assert body["fusion_method"] == "rrf"
        assert body["reranker_provider"] == "cross_encoder"
        assert body["reranker_enabled"] is True
    finally:
        app.dependency_overrides.pop(get_config, None)


def test_info_reflects_mcp_and_agent_toggles():
    """mcp.enabled/mcp.client.enabled/agent.enabled are reported independently."""
    config = load_config()
    agent = config.agent.model_copy(update={"enabled": True})
    mcp_client = config.mcp.client.model_copy(update={"enabled": True})
    mcp = config.mcp.model_copy(update={"enabled": True, "client": mcp_client})
    patched = config.model_copy(update={"agent": agent, "mcp": mcp})

    app.dependency_overrides[get_config] = lambda: patched
    try:
        client = TestClient(app)
        response = client.get("/info")
        assert response.status_code == 200
        body = response.json()
        assert body["agent_enabled"] is True
        assert body["mcp_enabled"] is True
        assert body["mcp_client_enabled"] is True
    finally:
        app.dependency_overrides.pop(get_config, None)


def test_info_never_exposes_secrets_or_infrastructure_details():
    """`GET /info` intentionally includes model names, but never a secret/URL/path/key.

    Contrasts with `GET /`'s own regression guard: `/info` exists
    precisely to expose provider/model identity, so it is checked against
    a narrower denylist (real secrets/infrastructure details), not against
    model-name substrings.
    """
    client = TestClient(app)
    response = client.get("/info")
    raw = response.text.lower()

    # The whole point of this endpoint: model identity IS present.
    assert "qwen" in raw or "sentence-transformers" in raw or "minilm" in raw

    for forbidden in (
        "postgresql://",
        "sqlite:///",
        "api_key",
        "secret",
        "bearer ",
        "authorization:",
        "c:\\",
        "/home/",
        "jwt",
    ):
        assert forbidden not in raw, f"GET /info leaked {forbidden!r}"
