"""Reranker interface: reorders (and optionally rescores) search results."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas import SearchResult


class Reranker(ABC):
    """Reorders (and optionally rescores) vector-search results."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Reorder (and optionally rescore) vector-search results.

        Parameters
        ----------
        query : str
            The original user query.
        results : list[SearchResult]
            Vector-search results to rerank, in their current order.
        top_n : int
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Up to `top_n` results, best-first.
        """
