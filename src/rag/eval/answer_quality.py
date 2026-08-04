"""Answer-quality scoring plug point.

KeywordOverlapScorer is a cheap placeholder so the eval CLI has something to
report today. Swap in an LLM-judge (e.g. via the same Ollama LLM used for
generation) by implementing AnswerQualityScorer — see README Roadmap."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AnswerQualityScorer(ABC):
    @abstractmethod
    def score(self, question: str, generated_answer: str, reference_answer: str | None = None) -> float:
        """Return a score in [0, 1]; higher is better."""


class KeywordOverlapScorer(AnswerQualityScorer):
    def score(self, question: str, generated_answer: str, reference_answer: str | None = None) -> float:
        if not reference_answer:
            return 0.0
        reference_words = {w.lower() for w in reference_answer.split() if len(w) > 3}
        if not reference_words:
            return 0.0
        generated_words = {w.lower() for w in generated_answer.split()}
        return len(reference_words & generated_words) / len(reference_words)
