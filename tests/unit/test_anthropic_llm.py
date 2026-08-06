from __future__ import annotations

import sys
import types
from typing import Any

from rag.generation.anthropic_llm import AnthropicLLM


class _FakeUsage:
    """Stand-in for an Anthropic message's usage object."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        """Store token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessages:
    """Stand-in for `client.messages`."""

    def __init__(self, response_text: str) -> None:
        """Store the fixed response text this double's create() will return."""
        self._response_text = response_text

    def create(self, **kwargs: Any) -> Any:
        """Return a fake message response with usage info."""
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(text=self._response_text)],
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )


class _FakeModels:
    """Stand-in for `client.models`."""

    def __init__(self, should_fail: bool) -> None:
        """Store whether list() should raise."""
        self._should_fail = should_fail

    def list(self) -> list[Any]:
        """Raise if configured to fail; else return an empty list."""
        if self._should_fail:
            raise RuntimeError("connection refused")
        return []


class _FakeAnthropicClient:
    """Stand-in for `anthropic.Anthropic(...)`."""

    def __init__(self, response_text: str = "fake answer", models_fail: bool = False) -> None:
        """Build fake `messages` and `models` sub-clients."""
        self.messages = _FakeMessages(response_text)
        self.models = _FakeModels(models_fail)


def _install_fake_anthropic_module(monkeypatch, client: _FakeAnthropicClient) -> None:
    """Inject a fake `anthropic` module into sys.modules so `import anthropic` finds it."""
    fake_module = types.SimpleNamespace(Anthropic=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_generate_returns_completion_text_and_tracks_usage(monkeypatch):
    """generate() returns the completion text and accumulates call/token usage."""
    _install_fake_anthropic_module(monkeypatch, _FakeAnthropicClient(response_text="hello world"))

    llm = AnthropicLLM(api_key="sk-ant-test")
    result = llm.generate("hi")

    assert result == "hello world"
    assert llm.call_count == 1
    assert llm.input_tokens == 10
    assert llm.output_tokens == 5

    llm.generate("hi again")
    assert llm.call_count == 2
    assert llm.input_tokens == 20
    assert llm.output_tokens == 10


def test_health_check_returns_false_on_exception(monkeypatch):
    """health_check() returns False when the underlying API call raises."""
    _install_fake_anthropic_module(monkeypatch, _FakeAnthropicClient(models_fail=True))

    llm = AnthropicLLM(api_key="sk-ant-test")
    assert llm.health_check() is False


def test_health_check_returns_true_when_reachable(monkeypatch):
    """health_check() returns True when the underlying API call succeeds."""
    _install_fake_anthropic_module(monkeypatch, _FakeAnthropicClient())

    llm = AnthropicLLM(api_key="sk-ant-test")
    assert llm.health_check() is True
