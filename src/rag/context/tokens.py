"""Token counting abstraction, decoupled from any provider's real tokenizer.

Neither Ollama nor this codebase's `LLM` ABC exposes a pre-call tokenizer:
`OllamaLLM` only learns `prompt_eval_count`/`eval_count` *after* a call
returns (see `generation/ollama_llm.py`). Context Engineering needs a
token estimate *before* the call, to decide what fits, so this module
documents and bounds that estimation gap explicitly rather than silently
treating a character count as a token count.

Two strategies:

- `"heuristic"` (the default, zero dependencies): a chars-per-token ratio
  calibrated for English prose (`config.context.token_counting.
  chars_per_token`, default 4.0, the commonly-cited rule of thumb for
  English text under BPE-style tokenizers). Deliberately conservative:
  rounds up, never down, since under-counting is the failure mode that
  causes a real overflow, while over-counting only wastes budget margin.
- `"tiktoken"` (optional, lazy-imported like `ragas`/`datasets`): uses
  OpenAI's `cl100k_base` encoding as a proxy. This is explicitly an
  approximation, not the real tokenizer for `qwen2.5`/any other Ollama
  model (those use their own BPE vocabularies); it is offered because it
  is a real sub-word tokenizer and therefore a tighter estimate than the
  character heuristic for most English text, not because it is exact.

`safety_margin_tokens` in `ContextBudget` exists specifically to absorb
the gap between either estimate and whatever `qwen2.5`'s own tokenizer
would have counted.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import AppConfig

logger = logging.getLogger(__name__)


class TokenCounter(ABC):
    """Estimates token counts for text that has not yet been sent to the LLM."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Return an estimated token count for `text`.

        Parameters
        ----------
        text : str
            Text to estimate. An empty string always counts as 0.

        Returns
        -------
        int
            A non-negative estimate. Implementations round up, never
            down, so a packing decision never under-budgets.
        """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """A short, fixed name for this strategy (for traces/metrics labels)."""


class HeuristicTokenCounter(TokenCounter):
    """Chars-per-token ratio estimator. No dependencies, always available."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """Construct the estimator.

        Parameters
        ----------
        chars_per_token : float, optional
            Average characters per token for the target language/text
            mix, by default 4.0 (English prose). Must be positive.
        """
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """See `TokenCounter.count`."""
        if not text:
            return 0
        return math.ceil(len(text) / self._chars_per_token)

    @property
    def strategy_name(self) -> str:
        """See `TokenCounter.strategy_name`."""
        return "heuristic"


class TiktokenTokenCounter(TokenCounter):
    """`tiktoken`-backed estimator (optional dependency, lazy-imported).

    Uses `cl100k_base` as a general-purpose sub-word tokenizer proxy.
    Not the real tokenizer for any model this codebase actually runs
    (`qwen2.5` via Ollama), see the module docstring.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Construct the estimator.

        Parameters
        ----------
        encoding_name : str, optional
            A `tiktoken` encoding name, by default ``"cl100k_base"``.

        Raises
        ------
        RuntimeError
            If the `tiktoken` package is not installed.
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "TiktokenTokenCounter requires the optional 'tiktoken' package. "
                "Install it (pip install tiktoken) or set "
                "context.token_counting.strategy: heuristic."
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """See `TokenCounter.count`."""
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    @property
    def strategy_name(self) -> str:
        """See `TokenCounter.strategy_name`."""
        return "tiktoken"


def build_token_counter(config: AppConfig) -> TokenCounter:
    """Construct the `TokenCounter` selected by `config.context.token_counting`.

    Falls back to `HeuristicTokenCounter` (logging a warning) if
    `strategy="tiktoken"` is configured but the package is not
    installed, since an unavailable tokenizer must never crash context
    assembly; it degrades to the always-available estimator instead.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    TokenCounter
        The constructed counter.
    """
    cfg = config.context.token_counting
    if cfg.strategy == "tiktoken":
        try:
            return TiktokenTokenCounter(cfg.tiktoken_encoding)
        except RuntimeError:
            logger.warning(
                "context.token_counting.strategy='tiktoken' but tiktoken is not "
                "installed; falling back to the heuristic counter",
                exc_info=True,
            )
    return HeuristicTokenCounter(cfg.chars_per_token)
