"""`LLM` backed by a local Ollama server."""

from __future__ import annotations

from typing import Any

import ollama

from rag.generation.base import LLM


class OllamaLLM(LLM):
    """Generates completions via a local (or remote) Ollama server.

    Tracks `last_prompt_tokens`/`last_completion_tokens` per call (not
    cumulative), read from Ollama's `prompt_eval_count`/`eval_count`
    response fields. `None` if either field is omitted, though this
    hasn't been observed in practice for a non-streaming call.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        seed: int | None = None,
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
        seed : int | None, optional
            Fixed sampling seed passed through to Ollama's `options.seed`
            for reproducible generation, by default `None` (omitted from
            `options` entirely, matching prior non-deterministic
            behavior. Not passed as a literal `0`, which is itself a
            valid, distinct seed value).
        """
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None

    def generate(self, system: str, user: str) -> str:
        """See `LLM.generate`.

        Uses Ollama's role-aware `chat()` endpoint (not the raw-completion
        `generate()` endpoint) so `system`/`user` reach the model as
        genuinely separate messages, not a concatenated string.
        """
        options: dict[str, Any] = {
            "temperature": self._temperature,
            "num_predict": self._max_tokens,
        }
        if self._seed is not None:
            options["seed"] = self._seed
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat(model=self._model_name, messages=messages, options=options)
        self.last_prompt_tokens = response.get("prompt_eval_count")
        self.last_completion_tokens = response.get("eval_count")
        return response["message"]["content"]

    def health_check(self) -> bool:
        """See `LLM.health_check`."""
        try:
            self._client.list()
            return True
        except Exception:
            return False
