from __future__ import annotations

import pytest

from rag.config import ChunkingConfig, load_config


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
