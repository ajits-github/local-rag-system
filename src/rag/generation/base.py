from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for a fully-formed prompt."""

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap connectivity check used by GET /health."""
