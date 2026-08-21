"""RAG_CONFIG_PATH override behavior for the API's get_config() singleton."""

from __future__ import annotations

import pytest

from rag.api.deps import get_config


@pytest.fixture(autouse=True)
def _clear_get_config_cache():
    """Isolate each test's process-wide get_config() singleton."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_get_config_defaults_to_default_yaml_when_env_unset(monkeypatch):
    """With RAG_CONFIG_PATH unset, get_config() loads config/default.yaml."""
    monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
    config = get_config()
    assert config.generation.model_name == "qwen2.5:1.5b"
    assert config.generation.seed is None


def test_get_config_empty_env_var_preserves_default_behavior(monkeypatch):
    """An empty RAG_CONFIG_PATH (e.g. Compose's `${RAG_CONFIG_PATH:-}`) is treated as unset."""
    monkeypatch.setenv("RAG_CONFIG_PATH", "")
    config = get_config()
    assert config.generation.model_name == "qwen2.5:1.5b"
    assert config.generation.seed is None


def test_get_config_loads_override_path_when_set(monkeypatch):
    """A non-empty RAG_CONFIG_PATH loads that file instead of the default."""
    monkeypatch.setenv("RAG_CONFIG_PATH", "config/experiments/classic-rag-baseline-v1.yaml")
    config = get_config()
    assert config.generation.seed == 42
    assert config.generation.temperature == 0.0


def test_get_config_caches_result_across_calls(monkeypatch):
    """get_config() stays a process-wide singleton regardless of which path it loaded."""
    monkeypatch.setenv("RAG_CONFIG_PATH", "config/experiments/classic-rag-baseline-v1.yaml")
    first = get_config()
    second = get_config()
    assert first is second


def test_get_config_raises_clearly_on_missing_override_path(monkeypatch):
    """An explicitly set but nonexistent RAG_CONFIG_PATH fails loudly, never falls back."""
    monkeypatch.setenv("RAG_CONFIG_PATH", "config/experiments/does-not-exist.yaml")
    with pytest.raises(RuntimeError, match="not found"):
        get_config()


def test_get_config_raises_clearly_on_unparseable_override_path(monkeypatch, tmp_path):
    """An explicitly set RAG_CONFIG_PATH pointing at invalid YAML fails loudly."""
    bad_config = tmp_path / "broken.yaml"
    bad_config.write_text("app: {name: [unterminated\n", encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG_PATH", str(bad_config))
    with pytest.raises(RuntimeError, match="not valid YAML"):
        get_config()
