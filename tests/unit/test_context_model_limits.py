"""Unit tests for `rag.context.model_limits`."""

from __future__ import annotations

from rag.config import load_config
from rag.context.model_limits import resolve_model_context_limits


def test_known_model_uses_its_override():
    """Known model uses its override."""
    config = load_config().model_copy(deep=True)
    config.generation.model_name = "qwen2.5:1.5b"
    limits = resolve_model_context_limits(config)
    assert limits.context_window_tokens == 32768
    assert limits.max_output_tokens == config.generation.max_tokens
    assert limits.default_reserved_output_tokens == config.generation.max_tokens


def test_unknown_model_falls_back_to_the_configured_default():
    """Unknown model falls back to the configured default."""
    config = load_config().model_copy(deep=True)
    config.generation.model_name = "some-model-nobody-added-an-override-for"
    config.context.model_limits.default_context_window_tokens = 8192
    limits = resolve_model_context_limits(config)
    assert limits.context_window_tokens == 8192
