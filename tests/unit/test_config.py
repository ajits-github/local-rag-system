from __future__ import annotations

import pytest

from rag.config import load_config


def test_load_default_config_has_expected_defaults():
    config = load_config()
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.generation.model_name == "qwen2.5:1.5b"
    assert config.vectorstore.connection_env_var == "DATABASE_URL"
    assert config.reranker.provider == "none"


def test_database_url_resolves_from_env(monkeypatch):
    config = load_config()
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    assert config.database_url() == "postgresql://rag:rag@localhost:15987/ragdb"


def test_database_url_raises_clear_error_when_unset(monkeypatch):
    config = load_config()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        config.database_url()


def test_ollama_base_url_has_sane_default(monkeypatch):
    config = load_config()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert config.ollama_base_url() == "http://localhost:11434"
