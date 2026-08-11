from __future__ import annotations

import pytest

from rag.config import ChunkingConfig, HybridRetrievalConfig, load_config


def test_load_default_config_has_expected_defaults():
    """config/default.yaml loads with its documented default values."""
    config = load_config()
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.generation.model_name == "qwen2.5:1.5b"
    assert config.vectorstore.connection_env_var == "DATABASE_URL"
    assert config.reranker.provider == "none"


def test_default_config_chunking_provider_is_structured_markdown():
    """config/default.yaml's chunking.provider defaults to structured_markdown with its tunables."""
    config = load_config()
    assert config.chunking.provider == "structured_markdown"
    assert config.chunking.structured_markdown.table_row_group_size == 20
    assert config.chunking.structured_markdown.max_atomic_block_chars == 2000


def test_chunking_config_structured_markdown_validates_with_defaults():
    """ChunkingConfig(provider='structured_markdown') validates with correct nested defaults."""
    config = ChunkingConfig(provider="structured_markdown")
    assert config.structured_markdown.table_row_group_size == 20
    assert config.structured_markdown.max_atomic_block_chars == 2000


def test_database_url_resolves_from_env(monkeypatch):
    """database_url() reads DATABASE_URL from the process environment."""
    config = load_config()
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    assert config.database_url() == "postgresql://rag:rag@localhost:15987/ragdb"


def test_database_url_raises_clear_error_when_unset(monkeypatch):
    """database_url() raises a RuntimeError naming the env var when unset."""
    config = load_config()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        config.database_url()


def test_ollama_base_url_has_sane_default(monkeypatch):
    """ollama_base_url() falls back to localhost when its env var is unset."""
    config = load_config()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert config.ollama_base_url() == "http://localhost:11434"


def test_judge_default_provider_is_not_a_generation_model():
    """The default judge provider/model never matches the generation model."""
    config = load_config()
    assert config.judge.provider == "openai"
    assert config.judge.ollama.model_name not in {"qwen2.5:1.5b", "qwen2.5:3b"}
    assert config.judge.ollama.model_name != config.generation.model_name


def test_openai_api_key_resolves_from_env(monkeypatch):
    """openai_api_key() reads OPENAI_API_KEY from the process environment."""
    config = load_config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    assert config.openai_api_key() == "sk-test-123"


def test_anthropic_api_key_resolves_from_env(monkeypatch):
    """anthropic_api_key() reads ANTHROPIC_API_KEY from the process environment."""
    config = load_config()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    assert config.anthropic_api_key() == "sk-ant-test-123"


def test_default_config_mlflow_is_enabled_with_local_sqlite_backend():
    """config/default.yaml enables MLflow by default, using a local SQLite-backed tracking store."""
    config = load_config()
    assert config.mlflow.enabled is True
    assert config.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert config.mlflow.experiment_name == "local-rag-system"


def test_default_config_retrieval_provider_is_dense():
    """config/default.yaml's retrieval.provider defaults to dense-only, hybrid tunables ready."""
    config = load_config()
    assert config.retrieval.provider == "dense"
    assert config.retrieval.hybrid.rrf_k == 60


def test_hybrid_retrieval_config_defaults():
    """HybridRetrievalConfig() has the standard RRF k=60 default."""
    assert HybridRetrievalConfig().rrf_k == 60


def test_retrieval_config_hybrid_validates_with_custom_rrf_k():
    """RetrievalConfig(provider='hybrid') accepts a custom nested rrf_k."""
    config = load_config().model_copy(deep=True)
    config.retrieval.provider = "hybrid"
    config.retrieval.hybrid.rrf_k = 30
    assert config.retrieval.provider == "hybrid"
    assert config.retrieval.hybrid.rrf_k == 30


def test_default_config_relationship_expansion_disabled_by_default():
    """config/default.yaml's relationship_expansion is off, a no-op unless explicitly enabled."""
    config = load_config()
    expansion = config.retrieval.relationship_expansion
    assert expansion.enabled is False
    assert expansion.include_parent is True
    assert expansion.include_neighbors is True
    assert expansion.max_related_elements == 3


def test_default_config_vision_provider_is_none():
    """config/default.yaml's vision.provider defaults to 'none' -- no image bytes ever read."""
    config = load_config()
    assert config.vision.provider == "none"


def test_multimodal_v2_text_only_experiment_config_loads_with_expected_overrides():
    """The Experiment A config activates prompt v2 + hybrid retrieval, expansion off."""
    config = load_config("config/experiments/multimodal-v2-text-only.yaml")
    assert config.generation.prompt.version == "v2"
    assert config.retrieval.provider == "hybrid"
    assert config.reranker.provider == "none"
    assert config.retrieval.relationship_expansion.enabled is False
    assert config.vision.provider == "none"
    assert config.generation.model_name == "qwen2.5:1.5b"


def test_multimodal_v2_relationship_experiment_config_only_differs_by_expansion():
    """The Experiment B config matches Experiment A except relationship_expansion.enabled=True."""
    config_a = load_config("config/experiments/multimodal-v2-text-only.yaml")
    config_b = load_config("config/experiments/multimodal-v2-relationship.yaml")
    assert config_b.retrieval.relationship_expansion.enabled is True
    a_dict = config_a.model_dump(exclude={"retrieval": {"relationship_expansion": {"enabled"}}})
    b_dict = config_b.model_dump(exclude={"retrieval": {"relationship_expansion": {"enabled"}}})
    assert a_dict == b_dict
