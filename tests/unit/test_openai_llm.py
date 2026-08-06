from __future__ import annotations

import sys
import types
from typing import Any

from rag.generation.openai_llm import OpenAILLM


class _FakeUsage:
    """Stand-in for an OpenAI chat completion's usage object."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Store token counts."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChoice:
    """Stand-in for one OpenAI chat completion choice."""

    def __init__(self, content: str) -> None:
        """Store the completion text."""
        self.message = types.SimpleNamespace(content=content)


class _FakeCompletions:
    """Stand-in for `client.chat.completions`."""

    def __init__(self, response_text: str) -> None:
        """Store the fixed response text this double's create() will return."""
        self._response_text = response_text

    def create(self, **kwargs: Any) -> Any:
        """Return a fake completion response with usage info."""
        return types.SimpleNamespace(
            choices=[_FakeChoice(self._response_text)],
            usage=_FakeUsage(prompt_tokens=10, completion_tokens=5),
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


class _FakeOpenAIClient:
    """Stand-in for `openai.OpenAI(...)`."""

    def __init__(self, response_text: str = "fake answer", models_fail: bool = False) -> None:
        """Build fake `chat.completions` and `models` sub-clients."""
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(response_text))
        self.models = _FakeModels(models_fail)


def _install_fake_openai_module(monkeypatch, client: _FakeOpenAIClient) -> None:
    """Inject a fake `openai` module into sys.modules so `import openai` finds it."""
    fake_module = types.SimpleNamespace(OpenAI=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_generate_returns_completion_text_and_tracks_usage(monkeypatch):
    """generate() returns the completion text and accumulates call/token usage."""
    _install_fake_openai_module(monkeypatch, _FakeOpenAIClient(response_text="hello world"))

    llm = OpenAILLM(api_key="sk-test")
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
    _install_fake_openai_module(monkeypatch, _FakeOpenAIClient(models_fail=True))

    llm = OpenAILLM(api_key="sk-test")
    assert llm.health_check() is False


def test_health_check_returns_true_when_reachable(monkeypatch):
    """health_check() returns True when the underlying API call succeeds."""
    _install_fake_openai_module(monkeypatch, _FakeOpenAIClient())

    llm = OpenAILLM(api_key="sk-test")
    assert llm.health_check() is True
