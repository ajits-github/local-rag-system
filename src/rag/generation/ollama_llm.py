"""`LLM` backed by a local Ollama server."""

from __future__ import annotations

import threading
from typing import Any

import ollama

from rag.generation.base import LLM


class OllamaLLM(LLM):
    """Generates completions via a local (or remote) Ollama server.

    Tracks `last_prompt_tokens`/`last_completion_tokens` per call (not
    cumulative), read from Ollama's `prompt_eval_count`/`eval_count`
    response fields. `None` if either field is omitted, though this
    hasn't been observed in practice for a non-streaming call.

    This one `OllamaLLM` instance is a process-wide singleton
    (`rag.api.deps.get_llm`), shared across concurrent requests. Storing
    `last_prompt_tokens`/`last_completion_tokens` as plain instance
    attributes would let two concurrent `generate()` calls race: request
    A's `generate()` returns, but before A's caller reads
    `last_prompt_tokens`, request B's `generate()` call (on a different
    worker thread) overwrites it with B's own counts, so A reports B's
    usage. Backed by `threading.local()` instead, so each calling
    thread sees only the counts from its own most recent call on this
    instance; every existing call site's `llm.last_prompt_tokens`/
    `getattr(llm, "last_prompt_tokens", None)` access is unchanged,
    since attribute syntax is identical either way.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
        seed: int | None = None,
        num_ctx: int | None = None,
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
        num_ctx : int | None, optional
            The model's real runtime context window, passed through to
            Ollama's `options.num_ctx`. `None` (omitted from `options`
            entirely) leaves Ollama's own default in effect, which is a
            real gap, not a safe fallback: Ollama defaults every model to
            a 2048-token context unless `num_ctx` is set explicitly,
            independent of that model's trained context length.
            `rag.factory.build_llm` resolves this from
            `rag.context.model_limits.resolve_model_context_limits`, the
            same table `rag.context.planner` uses to compute the Context
            Engineering token budget, so the two can never silently
            disagree about how large a prompt the real Ollama call will
            actually accept.
        """
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._num_ctx = num_ctx
        self._thread_local = threading.local()

    @property
    def last_prompt_tokens(self) -> int | None:
        """The calling thread's own most recent `generate()` call's prompt-token count."""
        return getattr(self._thread_local, "prompt_tokens", None)

    @property
    def last_completion_tokens(self) -> int | None:
        """The calling thread's own most recent `generate()` call's completion-token count."""
        return getattr(self._thread_local, "completion_tokens", None)

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
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat(model=self._model_name, messages=messages, options=options)
        self._thread_local.prompt_tokens = response.get("prompt_eval_count")
        self._thread_local.completion_tokens = response.get("eval_count")
        return response["message"]["content"]

    def health_check(self) -> bool:
        """See `LLM.health_check`."""
        try:
            self._client.list()
            return True
        except Exception:
            return False
