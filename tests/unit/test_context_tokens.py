"""Unit tests for `rag.context.tokens`."""

from __future__ import annotations

import pytest

from rag.context.tokens import HeuristicTokenCounter, TiktokenTokenCounter, build_token_counter


def test_heuristic_counter_rounds_up_and_handles_empty_text():
    """Heuristic counter rounds up and handles empty text."""
    counter = HeuristicTokenCounter(chars_per_token=4.0)
    assert counter.count("") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2  # rounds up, never down
    assert counter.strategy_name == "heuristic"


def test_heuristic_counter_rejects_non_positive_ratio():
    """Heuristic counter rejects non positive ratio."""
    with pytest.raises(ValueError):
        HeuristicTokenCounter(chars_per_token=0)


def test_build_token_counter_defaults_to_heuristic(monkeypatch):
    """Build token counter defaults to heuristic."""
    from rag.config import load_config

    config = load_config().model_copy(deep=True)
    counter = build_token_counter(config)
    assert counter.strategy_name == "heuristic"


def test_build_token_counter_falls_back_when_tiktoken_unavailable(monkeypatch):
    """Build token counter falls back when tiktoken unavailable."""
    import sys

    from rag.config import load_config

    config = load_config().model_copy(deep=True)
    config.context.token_counting.strategy = "tiktoken"
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = build_token_counter(config)
    # Falls back rather than raising.
    assert counter.strategy_name == "heuristic"


def test_tiktoken_counter_requires_the_optional_package(monkeypatch):
    """Tiktoken counter requires the optional package."""
    import sys

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(RuntimeError):
        TiktokenTokenCounter()
