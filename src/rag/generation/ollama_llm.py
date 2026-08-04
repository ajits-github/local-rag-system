from __future__ import annotations

import ollama

from rag.generation.base import LLM


class OllamaLLM(LLM):
    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self._client.generate(
            model=self._model_name,
            prompt=prompt,
            options={"temperature": self._temperature, "num_predict": self._max_tokens},
        )
        return response["response"]

    def health_check(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False
