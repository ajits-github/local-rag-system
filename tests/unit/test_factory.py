from __future__ import annotations

import pytest

from rag.config import load_config
from rag.factory import build_judge_llm, build_reranker
from rag.generation.anthropic_llm import AnthropicLLM
from rag.generation.ollama_llm import OllamaLLM
from rag.generation.openai_llm import OpenAILLM
from rag.rerankers.noop import NoOpReranker


def test_build_reranker_none_provider_returns_noop():
    """The default 'none' provider builds a NoOpReranker."""
    config = load_config()
    assert config.reranker.provider == "none"
    assert isinstance(build_reranker(config), NoOpReranker)


def test_build_reranker_cohere_without_api_key_raises(monkeypatch):
    """Selecting 'cohere' without its API key set raises a clear RuntimeError."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    config = load_config().model_copy(deep=True)
    config.reranker.provider = "cohere"

    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        build_reranker(config)


def test_build_judge_llm_ollama_provider_returns_ollama_llm():
    """The 'ollama' judge provider builds an OllamaLLM using judge.ollama's model."""
    config = load_config().model_copy(deep=True)
    config.judge.provider = "ollama"
    judge = build_judge_llm(config)
    assert isinstance(judge, OllamaLLM)


def test_build_judge_llm_openai_without_api_key_raises(monkeypatch):
    """Selecting 'openai' without its API key set raises a clear RuntimeError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_config().model_copy(deep=True)
    config.judge.provider = "openai"

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_judge_llm(config)


def test_build_judge_llm_openai_with_api_key_returns_openai_llm(monkeypatch):
    """Selecting 'openai' with its API key set builds an OpenAILLM."""
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    config = load_config().model_copy(deep=True)
    config.judge.provider = "openai"
    judge = build_judge_llm(config)
    assert isinstance(judge, OpenAILLM)


def test_build_judge_llm_anthropic_without_api_key_raises(monkeypatch):
    """Selecting 'anthropic' without its API key set raises a clear RuntimeError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config().model_copy(deep=True)
    config.judge.provider = "anthropic"

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_judge_llm(config)


def test_build_judge_llm_anthropic_with_api_key_returns_anthropic_llm(monkeypatch):
    """Selecting 'anthropic' with its API key set builds an AnthropicLLM."""
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    config = load_config().model_copy(deep=True)
    config.judge.provider = "anthropic"
    judge = build_judge_llm(config)
    assert isinstance(judge, AnthropicLLM)


def test_build_judge_llm_unknown_provider_raises_value_error():
    """An unrecognized judge.provider raises ValueError."""
    config = load_config().model_copy(deep=True)
    config.judge.provider = "not-a-real-provider"

    with pytest.raises(ValueError, match="Unknown judge provider"):
        build_judge_llm(config)
