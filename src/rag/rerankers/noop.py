"""Default `Reranker`: identity passthrough."""

from __future__ import annotations

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class NoOpReranker(Reranker):
    """Default reranker: identity passthrough, zero added latency."""

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Truncate `results` to `top_n` without reordering or rescoring."""
        return results[:top_n]
