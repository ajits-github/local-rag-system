"""`LLM` backed by the OpenAI chat completions API.

Not used as a `generation.provider` (still `ollama`-only) — wired in only
via `factory.build_judge_llm` for RAGAS judging. Requires the `ragas`
extra (which pulls in the `openai` package transitively): pip install .[ragas]
"""

from __future__ import annotations

from rag.generation.base import LLM


class OpenAILLM(LLM):
    """Generates completions via the OpenAI chat completions API.

    Tracks `call_count`/`input_tokens`/`output_tokens` itself rather than
    relying on ragas's `token_usage_parser`, whose compatibility with a
    custom (non-LangChain-native) LLM wrapper is unconfirmed.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        """Construct the OpenAI client.

        Parameters
        ----------
        api_key : str
            OpenAI API key.
        model_name : str, optional
            OpenAI chat model id, by default ``"gpt-4o-mini"``.
        temperature : float, optional
            Sampling temperature, by default 0.0.
        max_tokens : int, optional
            Maximum tokens to generate, by default 1024.

        Raises
        ------
        RuntimeError
            If the optional ``openai`` package isn't installed.
        """
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "judge.provider is 'openai' but the 'openai' package isn't installed. "
                "Install it with: pip install .[ragas]"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, system: str, user: str) -> str:
        """See `LLM.generate`."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        self.call_count += 1
        if response.usage is not None:
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""

    def health_check(self) -> bool:
        """See `LLM.health_check`."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
