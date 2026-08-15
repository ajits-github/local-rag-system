"""LLM interface: generates a completion from role-separated prompt turns."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    """Generates a completion from a system instruction turn and a user turn.

    `system`/`user` are kept as two distinct arguments (not
    pre-concatenated into one string) so every implementation can deliver
    them to the underlying model as genuinely separate channels -- e.g.
    Ollama's/OpenAI's/Anthropic's own role-aware chat APIs -- rather than
    relying on prompt-text labeling alone to keep system instructions,
    retrieved evidence, and the user's question apart (see
    `retrieval/pipeline.py`'s `answer()` and the auth-boundary milestone's
    prompt-injection-architecture hardening).
    """

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate a completion for a role-separated prompt.

        Parameters
        ----------
        system : str
            System instruction text (may be empty). Never contains
            retrieved content.
        user : str
            User-turn text -- the rendered retrieved context plus the
            question, per the active prompt template.

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
