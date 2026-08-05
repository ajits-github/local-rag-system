"""LLM interface: generates a completion for a fully-formed prompt."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    """Generates a completion for a fully-formed prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for a fully-formed prompt.

        Parameters
        ----------
        prompt : str
            Fully-formed prompt, including any retrieved context.

        Returns
        -------
        str
            The generated completion text.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap connectivity check used by GET /health.

        Returns
        -------
        bool
            True if the backend is reachable.
        """
