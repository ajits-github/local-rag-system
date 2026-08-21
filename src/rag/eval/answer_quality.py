"""Answer-quality scoring plug point.

KeywordOverlapScorer is a cheap placeholder so the eval CLI has something to
report today. Swap in an LLM-judge (e.g. via the same Ollama LLM used for
generation) by implementing AnswerQualityScorer. See README Roadmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AnswerQualityScorer(ABC):
    """Scores a generated answer against a question (and optional reference)."""

    @abstractmethod
    def score(
        self, question: str, generated_answer: str, reference_answer: str | None = None
    ) -> float:
        """Return a score in [0, 1]; higher is better.

        Parameters
        ----------
        question : str
            The original user question.
        generated_answer : str
            The LLM's generated answer.
        reference_answer : str | None, optional
            The gold expected answer, if available.

        Returns
        -------
        float
            A score in ``[0, 1]``.
        """


class KeywordOverlapScorer(AnswerQualityScorer):
    """Cheap placeholder scorer based on reference-keyword overlap.

    Score is the fraction of reference keywords (len > 3) that also
    appear in the generated answer.
    """

    def score(
        self, question: str, generated_answer: str, reference_answer: str | None = None
    ) -> float:
        """See `AnswerQualityScorer.score`.

        Returns 0.0 when `reference_answer` is missing or has no words
        longer than 3 characters to compare against.
        """
        if not reference_answer:
            return 0.0
        reference_words = {w.lower() for w in reference_answer.split() if len(w) > 3}
        if not reference_words:
            return 0.0
        generated_words = {w.lower() for w in generated_answer.split()}
        return len(reference_words & generated_words) / len(reference_words)
