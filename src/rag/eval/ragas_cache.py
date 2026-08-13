"""Persistent, judge-identity-safe caching for RAGAS judge LLM calls.

RAGAS ships its own disk cache (`ragas.cache.DiskCacheBackend` /
`CacheInterface`, wired in via `cache=` on `LangchainLLMWrapper`): every
`generate_text`/`agenerate_text` call is memoized under a key hashed from
the function name plus its call args/kwargs (`ragas.cache._generate_cache_key`).
Because that hash is computed from a *bound* method, the wrapped LLM
instance itself (and therefore the judge provider/model it was built from)
never enters the hash -- only the rendered prompt text and generation
kwargs (`n`/`temperature`/`stop`) do. That's fine for detecting a changed
question/answer/context/metric, since RAGAS's own metric prompts (see
e.g. `ragas.metrics._faithfulness.NLIStatementPrompt`) render the full
question/context/statements/instruction text into that prompt -- but it
means an unmodified `DiskCacheBackend` shared across two different judge
models would silently replay one model's verdict for another model's
byte-identical-looking prompt. That's exactly the kind of
experiment-invalidating bug this project has hit before with a
similarly "looks harmless, isn't" config coupling (see CLAUDE.md's
`rerank_top_n` story) -- so this module namespaces every cache key by an
explicit fingerprint of the judge's provider/model/temperature/max_tokens
(plus the installed `ragas` version, so a RAGAS upgrade that changes a
metric's prompt template also invalidates the cache) before delegating to
RAGAS's own disk-backed storage.

`reference_contexts` (this project's gold-authored evidence-level ground
truth, see CLAUDE.md's "Multimodal ingestion" section) is deliberately
not part of this module's key material: `ragas_scorer.build_dataset` never
feeds it into the RAGAS dataset (only `question`/`answer`/`contexts`/
`expected_answer` are), so it isn't an input to any judged metric today --
nothing to key on. If a future change starts passing it to RAGAS, it will
already be covered by the same mechanism that covers `contexts` today,
since it will become part of the rendered judge prompt text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.config import AppConfig

CACHE_SCHEMA_VERSION = "v1"

# Confirmed against this project's own tracked judge usage (see
# PROJECT_JOURNAL.md's experiment_015 entry: 1,115,731 input / 101,857
# output tokens on gpt-4o-mini priced out to $0.2285, matching OpenAI's
# published $0.15/1M input, $0.60/1M output). Deliberately not populated
# for models this project hasn't verified a real cost against -- see
# `estimate_avoided_cost`'s "no pricing configured" branch rather than
# guessing a number.
PRICING_USD_PER_1M_TOKENS: dict[tuple[str, str], dict[str, float]] = {
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
}


@dataclass
class CacheStats:
    """Hit/miss counters for one `NamespacedDiskCache` instance."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        """Total lookups recorded (hits + misses)."""
        return self.hits + self.misses

    def as_dict(self) -> dict[str, int]:
        """Return `{"hits", "misses", "total"}`."""
        return {"hits": self.hits, "misses": self.misses, "total": self.total}


def _ragas_version() -> str:
    """Best-effort installed `ragas` version, or `"unknown"` if unavailable."""
    try:
        import ragas
    except ImportError:
        return "unknown"
    return getattr(ragas, "__version__", "unknown")


def judge_fingerprint(config: AppConfig) -> str:
    """Build a short, stable fingerprint of the active judge configuration.

    Used as a cache-key namespace so a cache entry produced by one judge
    provider/model can never be replayed for another -- see this module's
    docstring for why RAGAS's own key hash doesn't already guarantee that.

    Parameters
    ----------
    config : AppConfig
        Application configuration; only `config.judge` is read.

    Returns
    -------
    str
        A short hex digest deterministic in
        (schema version, ragas version, judge provider, judge model,
        judge temperature, judge max_tokens).
    """
    judge = config.judge
    model_name = {
        "openai": judge.openai.model_name,
        "anthropic": judge.anthropic.model_name,
        "ollama": judge.ollama.model_name,
    }[judge.provider]
    raw = "|".join(
        [
            CACHE_SCHEMA_VERSION,
            _ragas_version(),
            judge.provider,
            model_name,
            f"{judge.temperature}",
            f"{judge.max_tokens}",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class NamespacedDiskCache:
    """A `ragas.cache.CacheInterface` that prefixes every key with a namespace.

    Delegates actual storage to `ragas.cache.DiskCacheBackend`; the only
    behavior added is (a) namespacing every key so different namespaces
    never collide even given an identical underlying RAGAS-computed hash,
    and (b) counting hits/misses (`self.stats`) so callers can report
    avoided calls/cost. Constructed with `build_judge_cache` in normal use.
    """

    def __init__(self, cache_dir: str | Path, namespace: str) -> None:
        """Open (or create) the on-disk cache and set the key namespace.

        Parameters
        ----------
        cache_dir : str | Path
            Directory RAGAS's `DiskCacheBackend` persists entries under.
        namespace : str
            Prefix applied to every key (see `judge_fingerprint`).

        Raises
        ------
        RuntimeError
            If the `ragas` package isn't installed.
        """
        try:
            from ragas.cache import DiskCacheBackend
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS judge-call caching requires the 'ragas' extra. "
                "Install with: pip install .[ragas]"
            ) from exc
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._backend = DiskCacheBackend(cache_dir=str(cache_dir))
        self._namespace = namespace
        self.stats = CacheStats()

    def _namespaced(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> Any:
        """See `ragas.cache.CacheInterface.get`."""
        return self._backend.get(self._namespaced(key))

    def set(self, key: str, value: Any) -> None:
        """See `ragas.cache.CacheInterface.set`."""
        self._backend.set(self._namespaced(key), value)

    def has_key(self, key: str) -> bool:
        """See `ragas.cache.CacheInterface.has_key`; also records a hit/miss."""
        hit = self._backend.has_key(self._namespaced(key))
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return hit


def build_judge_cache(config: AppConfig) -> NamespacedDiskCache:
    """Build a `NamespacedDiskCache` for the currently configured judge.

    Parameters
    ----------
    config : AppConfig
        Application configuration; reads `config.judge.cache_dir` and
        everything `judge_fingerprint` reads.

    Returns
    -------
    NamespacedDiskCache
        A cache namespaced to this exact judge provider/model/config, so
        switching either one is guaranteed to miss rather than silently
        replay another judge's verdict.
    """
    return NamespacedDiskCache(
        cache_dir=config.judge.cache_dir, namespace=judge_fingerprint(config)
    )


def estimate_avoided_cost(
    provider: str,
    model_name: str,
    avoided_calls: int,
    judge_llm: Any,
) -> dict[str, Any] | None:
    """Estimate the USD cost avoided by `avoided_calls` cache hits this run.

    Extrapolates from the *same run's* actual (uncached) judge usage --
    `judge_llm.call_count`/`.input_tokens`/`.output_tokens`, tracked by
    `OpenAILLM`/`AnthropicLLM` themselves (see their docstrings) -- rather
    than a fixed per-call assumption, since prompt length varies by
    question/context. Returns `None` when there's nothing to estimate
    (`avoided_calls == 0`); otherwise returns a dict that always explains
    *why* if it can't produce a number (local provider, no uncached calls
    to extrapolate from this run, or no pricing on file for this model) --
    never a guessed dollar figure.

    Parameters
    ----------
    provider : str
        `config.judge.provider` (`"openai"`/`"anthropic"`/`"ollama"`).
    model_name : str
        The judge's configured model name.
    avoided_calls : int
        Number of cache hits this run (`NamespacedDiskCache.stats.hits`).
    judge_llm : Any
        The judge `LLM` instance used this run; read for
        `call_count`/`input_tokens`/`output_tokens` when present.

    Returns
    -------
    dict[str, Any] | None
        `None` if `avoided_calls <= 0`; otherwise a dict with either
        `estimated_cost_usd` (float) and supporting fields, or
        `estimated_cost_usd: None` plus a `reason` string.
    """
    if avoided_calls <= 0:
        return None
    if provider not in {"openai", "anthropic"}:
        return {
            "avoided_calls": avoided_calls,
            "estimated_cost_usd": None,
            "reason": f"provider '{provider}' is local/free -- no hosted API cost",
        }
    call_count = getattr(judge_llm, "call_count", 0)
    input_tokens = getattr(judge_llm, "input_tokens", 0)
    output_tokens = getattr(judge_llm, "output_tokens", 0)
    if call_count == 0:
        return {
            "avoided_calls": avoided_calls,
            "estimated_cost_usd": None,
            "reason": (
                "no uncached judge calls this run to derive an average "
                "token profile from (every call was a cache hit)"
            ),
        }
    pricing = PRICING_USD_PER_1M_TOKENS.get((provider, model_name))
    if pricing is None:
        return {
            "avoided_calls": avoided_calls,
            "estimated_cost_usd": None,
            "reason": (
                f"no pricing on file for ({provider}, {model_name}) in "
                "ragas_cache.PRICING_USD_PER_1M_TOKENS"
            ),
        }
    avg_input = input_tokens / call_count
    avg_output = output_tokens / call_count
    estimated_cost = (
        avg_input * avoided_calls * pricing["input"] / 1_000_000
        + avg_output * avoided_calls * pricing["output"] / 1_000_000
    )
    return {
        "avoided_calls": avoided_calls,
        "avg_input_tokens_per_call": round(avg_input, 1),
        "avg_output_tokens_per_call": round(avg_output, 1),
        "estimated_cost_usd": round(estimated_cost, 4),
        "note": (
            "approximate: assumes avoided calls have the same average "
            "token profile as the calls actually made this run"
        ),
    }
