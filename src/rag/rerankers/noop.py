from __future__ import annotations

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class NoOpReranker(Reranker):
    """Default reranker: identity passthrough, zero added latency."""

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        return results[:top_n]
