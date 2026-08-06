"""`LLM` backed by the Anthropic messages API.

Not used as a `generation.provider` (still `ollama`-only) — wired in only
via `factory.build_judge_llm` for RAGAS judging. Requires the `anthropic`
extra: pip install .[anthropic]
"""

from __future__ import annotations

from rag.generation.base import LLM


class AnthropicLLM(LLM):
    """Generates completions via the Anthropic messages API.

    Tracks `call_count`/`input_tokens`/`output_tokens` itself rather than
    relying on ragas's `token_usage_parser`, whose compatibility with a
    custom (non-LangChain-native) LLM wrapper is unconfirmed.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Construct the Anthropic client.

        Parameters
        ----------
        api_key : str
            Anthropic API key.
        model_name : str, optional
            Anthropic model id, by default ``"claude-haiku-4-5-20251001"``.
        temperature : float, optional
            Sampling temperature, by default 0.0.
        max_tokens : int, optional
            Maximum tokens to generate, by default 1024.

        Raises
        ------
        RuntimeError
            If the optional ``anthropic`` package isn't installed.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "judge.provider is 'anthropic' but the 'anthropic' package isn't installed. "
                "Install it with: pip install .[anthropic]"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, prompt: str) -> str:
        """See `LLM.generate`."""
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        self.call_count += 1
        if response.usage is not None:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        return response.content[0].text if response.content else ""

    def health_check(self) -> bool:
        """See `LLM.health_check`."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
