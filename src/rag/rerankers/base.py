from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas import SearchResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Reorder (and optionally rescore) vector-search results."""
