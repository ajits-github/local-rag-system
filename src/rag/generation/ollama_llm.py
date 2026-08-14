"""`LLM` backed by a local Ollama server."""

from __future__ import annotations

import ollama

from rag.generation.base import LLM


class OllamaLLM(LLM):
    """Generates completions via a local (or remote) Ollama server.

    Tracks `last_prompt_tokens`/`last_completion_tokens` from Ollama's own
    `prompt_eval_count`/`eval_count` response fields (mirrors
    `OpenAILLM`'s `call_count`/`input_tokens`/`output_tokens` tracking,
    but per-call rather than cumulative, since production callers care
    about one question's token footprint, not a running total). `None`
    when Ollama's response omits either field (observed to always be
    present for a non-streaming `generate()` call, but not documented as
    guaranteed).
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        """Construct the Ollama client.

        Parameters
        ----------
        model_name : str, optional
            Ollama model tag, by default ``"qwen2.5:1.5b"``.
        base_url : str, optional
            Ollama server URL, by default ``"http://localhost:11434"``.
        temperature : float, optional
            Sampling temperature, by default 0.2.
        max_tokens : int, optional
            Maximum tokens to generate, by default 512.
        """
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def generate(self, prompt: str) -> str:
        """See `LLM.generate`."""
        response = self._client.generate(
            model=self._model_name,
            prompt=prompt,
            options={"temperature": self._temperature, "num_predict": self._max_tokens},
        )
        self.last_prompt_tokens = response.get("prompt_eval_count")
        self.last_completion_tokens = response.get("eval_count")
        return response["response"]

    def health_check(self) -> bool:
        """See `LLM.health_check`."""
        try:
            self._client.list()
            return True
        except Exception:
            return False
