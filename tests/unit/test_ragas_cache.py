"""Tests for `rag.eval.ragas_cache`: judge-identity-safe RAGAS judge-call caching.

Every test here uses local, in-process doubles only (a `_CountingLLM` that
never makes a network call, `tmp_path`-scoped disk caches) -- no hosted
judge API is ever reached, matching the "do not perform a paid RAGAS run
merely to test caching" constraint this module was built under.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ragas")  # every test below exercises the real ragas.cache machinery

from langchain_core.prompt_values import StringPromptValue  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.eval.ragas_adapters import LangchainLLMAdapter  # noqa: E402
from rag.eval.ragas_cache import (  # noqa: E402
    NamespacedDiskCache,
    build_judge_cache,
    estimate_avoided_cost,
    judge_fingerprint,
)
from rag.generation.base import LLM  # noqa: E402


class _CountingLLM(LLM):
    """LLM double recording every prompt string it's asked to answer."""

    def __init__(self) -> None:
        """Start with no recorded prompts."""
        self.prompts: list[str] = []

    def generate(self, system: str, user: str) -> str:
        """Record `user` and return a fixed response."""
        self.prompts.append(user)
        return "ok"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class _FakeJudgeUsage:
    """Duck-typed judge LLM double exposing only usage-tracking attributes."""

    def __init__(self, call_count: int, input_tokens: int, output_tokens: int) -> None:
        """Store fixed usage counters."""
        self.call_count = call_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _judge_config(provider: str = "openai", model_name: str = "gpt-4o-mini", cache_dir=None):
    """Build an AppConfig with `judge` overridden for a given provider/model/cache_dir."""
    config = load_config().model_copy(deep=True)
    config.judge.provider = provider
    if provider == "openai":
        config.judge.openai.model_name = model_name
    elif provider == "anthropic":
        config.judge.anthropic.model_name = model_name
    else:
        config.judge.ollama.model_name = model_name
    if cache_dir is not None:
        config.judge.cache_dir = str(cache_dir)
    return config


def _wrapped_llm(cache_dir, namespace: str, counting_llm: _CountingLLM):
    """Build a cache-wrapped `LangchainLLMWrapper` around `counting_llm`."""
    cache = NamespacedDiskCache(cache_dir=cache_dir, namespace=namespace)
    wrapper = LangchainLLMWrapper(LangchainLLMAdapter(rag_llm=counting_llm), cache=cache)
    return wrapper, cache


# -- judge_fingerprint -------------------------------------------------------


def test_judge_fingerprint_differs_by_model():
    """Two configs differing only in judge model get different fingerprints."""
    a = judge_fingerprint(_judge_config(model_name="gpt-4o-mini"))
    b = judge_fingerprint(_judge_config(model_name="gpt-4o"))
    assert a != b


def test_judge_fingerprint_differs_by_provider():
    """Two configs differing only in judge provider get different fingerprints."""
    a = judge_fingerprint(_judge_config(provider="openai", model_name="gpt-4o-mini"))
    b = judge_fingerprint(_judge_config(provider="anthropic", model_name="gpt-4o-mini"))
    assert a != b


def test_judge_fingerprint_stable_for_identical_config():
    """The same judge config always produces the same fingerprint."""
    a = judge_fingerprint(_judge_config())
    b = judge_fingerprint(_judge_config())
    assert a == b


# -- NamespacedDiskCache / cacher() integration ------------------------------


def test_identical_inputs_hit_cache_on_second_call(tmp_path):
    """A byte-identical second call is served from cache, not the judge LLM."""
    llm = _CountingLLM()
    wrapper, cache = _wrapped_llm(tmp_path, "ns-a", llm)
    prompt = StringPromptValue(text="question=X answer=Y context=Z")

    wrapper.generate_text(prompt=prompt)
    wrapper.generate_text(prompt=prompt)

    assert len(llm.prompts) == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_changed_generated_answer_misses_cache(tmp_path):
    """A prompt whose embedded generated answer differs is a cache miss."""
    llm = _CountingLLM()
    wrapper, cache = _wrapped_llm(tmp_path, "ns-a", llm)

    wrapper.generate_text(prompt=StringPromptValue(text="question=X answer=Y context=Z"))
    wrapper.generate_text(
        prompt=StringPromptValue(text="question=X answer=SOMETHING-ELSE context=Z")
    )

    assert len(llm.prompts) == 2
    assert cache.stats.misses == 2
    assert cache.stats.hits == 0


def test_changed_retrieved_context_misses_cache(tmp_path):
    """A prompt whose embedded retrieved context differs is a cache miss."""
    llm = _CountingLLM()
    wrapper, cache = _wrapped_llm(tmp_path, "ns-a", llm)

    wrapper.generate_text(prompt=StringPromptValue(text="question=X answer=Y context=Z"))
    wrapper.generate_text(
        prompt=StringPromptValue(text="question=X answer=Y context=SOMETHING-ELSE")
    )

    assert len(llm.prompts) == 2
    assert cache.stats.misses == 2
    assert cache.stats.hits == 0


def test_changed_judge_model_misses_cache_even_with_identical_prompt(tmp_path):
    """Namespacing by judge fingerprint means a model swap always misses.

    Both wrappers share the same underlying cache directory (simulating
    `config.judge.cache_dir` staying fixed across runs) -- only the
    namespace (derived from `judge_fingerprint`) differs, proving the
    isolation comes from the namespace, not from separate storage.
    """
    prompt = StringPromptValue(text="question=X answer=Y context=Z")

    llm_a = _CountingLLM()
    wrapper_a, _ = _wrapped_llm(
        tmp_path, judge_fingerprint(_judge_config(model_name="gpt-4o-mini")), llm_a
    )
    wrapper_a.generate_text(prompt=prompt)

    llm_b = _CountingLLM()
    wrapper_b, cache_b = _wrapped_llm(
        tmp_path, judge_fingerprint(_judge_config(model_name="gpt-4o")), llm_b
    )
    wrapper_b.generate_text(prompt=prompt)

    assert len(llm_a.prompts) == 1
    assert len(llm_b.prompts) == 1  # not served from llm_a's cached entry
    assert cache_b.stats.misses == 1
    assert cache_b.stats.hits == 0


def test_build_judge_cache_uses_config_cache_dir_and_starts_empty(tmp_path):
    """build_judge_cache reads config.judge.cache_dir and starts with zero stats."""
    config = _judge_config(cache_dir=tmp_path / "ragas-cache")
    cache = build_judge_cache(config)

    assert isinstance(cache, NamespacedDiskCache)
    assert cache.stats.as_dict() == {"hits": 0, "misses": 0, "total": 0}


# -- estimate_avoided_cost ----------------------------------------------------


def test_estimate_avoided_cost_none_when_nothing_avoided():
    """No estimate is produced when there were no cache hits to avoid a call for."""
    assert estimate_avoided_cost("openai", "gpt-4o-mini", 0, judge_llm=object()) is None


def test_estimate_avoided_cost_local_provider_has_no_cost():
    """A local (ollama) judge has no hosted cost, and says so instead of guessing."""
    judge = _FakeJudgeUsage(call_count=5, input_tokens=100, output_tokens=50)

    result = estimate_avoided_cost("ollama", "llama3.1:8b", 3, judge_llm=judge)

    assert result["estimated_cost_usd"] is None
    assert "reason" in result


def test_estimate_avoided_cost_no_uncached_calls_explains_itself():
    """A 100%-cache-hit run can't estimate cost (no average to extrapolate from)."""
    judge = _FakeJudgeUsage(call_count=0, input_tokens=0, output_tokens=0)

    result = estimate_avoided_cost("openai", "gpt-4o-mini", 5, judge_llm=judge)

    assert result["estimated_cost_usd"] is None
    assert "reason" in result


def test_estimate_avoided_cost_unpriced_model_explains_itself():
    """A model with no entry in the pricing table reports why, not a guessed number."""
    judge = _FakeJudgeUsage(call_count=5, input_tokens=100, output_tokens=50)

    result = estimate_avoided_cost("openai", "some-future-model", 3, judge_llm=judge)

    assert result["estimated_cost_usd"] is None
    assert "reason" in result


def test_estimate_avoided_cost_computes_from_this_runs_average():
    """A priced model with real uncached usage this run gets a computed estimate."""
    judge = _FakeJudgeUsage(call_count=10, input_tokens=1000, output_tokens=500)

    result = estimate_avoided_cost("openai", "gpt-4o-mini", 10, judge_llm=judge)

    # avg 100 input / 50 output tokens per call, x10 avoided calls, at
    # $0.15/1M input + $0.60/1M output (see PRICING_USD_PER_1M_TOKENS).
    expected = (10 * 100 * 0.15 / 1_000_000) + (10 * 50 * 0.60 / 1_000_000)
    assert result["avoided_calls"] == 10
    assert result["estimated_cost_usd"] == round(expected, 4)
