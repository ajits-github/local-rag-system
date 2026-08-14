from __future__ import annotations

from rag.generation.ollama_llm import OllamaLLM


class _FakeOllamaClient:
    """Stand-in for `ollama.Client`, mocked at the class boundary per project convention."""

    def __init__(self, response: dict, list_raises: bool = False) -> None:
        """Store the fixed `generate()` response this double will return."""
        self._response = response
        self._list_raises = list_raises
        self.last_options: dict | None = None

    def generate(self, model: str, prompt: str, options: dict) -> dict:
        """Return the fixed response, recording `options` for assertions."""
        self.last_options = options
        return self._response

    def list(self) -> None:
        """Raise if configured to simulate an unreachable server."""
        if self._list_raises:
            raise ConnectionError("unreachable")


def _make_llm(monkeypatch, response: dict, **kwargs) -> OllamaLLM:
    """Build an OllamaLLM whose internal ollama.Client is replaced with a fake."""
    llm = OllamaLLM(model_name="qwen2.5:3b", **kwargs)
    monkeypatch.setattr(llm, "_client", _FakeOllamaClient(response))
    return llm


def test_generate_captures_prompt_and_completion_token_counts(monkeypatch):
    """generate() maps prompt_eval_count/eval_count to last_prompt_tokens/last_completion_tokens."""
    llm = _make_llm(
        monkeypatch,
        {"response": "The answer.", "prompt_eval_count": 123, "eval_count": 45},
    )

    answer = llm.generate("some prompt")

    assert answer == "The answer."
    assert llm.last_prompt_tokens == 123
    assert llm.last_completion_tokens == 45


def test_generate_token_counts_are_none_when_response_omits_them(monkeypatch):
    """Token count attrs fall back to None, never raise, if Ollama's response omits them."""
    llm = _make_llm(monkeypatch, {"response": "The answer."})

    llm.generate("some prompt")

    assert llm.last_prompt_tokens is None
    assert llm.last_completion_tokens is None


def test_generate_updates_token_counts_across_successive_calls(monkeypatch):
    """Token count attrs reflect the most recent call only, not a running total."""
    llm = _make_llm(monkeypatch, {"response": "first", "prompt_eval_count": 10, "eval_count": 5})
    llm.generate("first prompt")
    assert llm.last_prompt_tokens == 10

    monkeypatch.setattr(
        llm,
        "_client",
        _FakeOllamaClient({"response": "second", "prompt_eval_count": 20, "eval_count": 8}),
    )
    llm.generate("second prompt")

    assert llm.last_prompt_tokens == 20
    assert llm.last_completion_tokens == 8


def test_generate_passes_seed_in_options_when_configured(monkeypatch):
    """A configured seed is forwarded to Ollama's options.seed."""
    llm = _make_llm(monkeypatch, {"response": "answer"}, seed=42)

    llm.generate("some prompt")

    assert llm._client.last_options["seed"] == 42


def test_generate_omits_seed_from_options_when_not_configured(monkeypatch):
    """No seed key is sent at all when seed is None (default), preserving non-determinism."""
    llm = _make_llm(monkeypatch, {"response": "answer"})

    llm.generate("some prompt")

    assert "seed" not in llm._client.last_options


def test_generate_passes_configured_temperature_and_max_tokens(monkeypatch):
    """temperature/num_predict are always present in options, independent of seed."""
    llm = _make_llm(monkeypatch, {"response": "answer"}, temperature=0.0, max_tokens=256)

    llm.generate("some prompt")

    assert llm._client.last_options["temperature"] == 0.0
    assert llm._client.last_options["num_predict"] == 256
