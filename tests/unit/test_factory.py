from __future__ import annotations

import pytest

from rag.config import load_config
from rag.factory import build_reranker
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
