"""Default `Reranker`: identity passthrough."""

from __future__ import annotations

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class NoOpReranker(Reranker):
    """Default reranker: a true identity, zero added latency.

    Ignores `top_n` entirely -- does not reorder, rescore, or truncate.
    The final number of chunks reaching generation is controlled solely by
    `retrieval.generation_context_top_n` in `RetrievalPipeline.retrieve()`,
    not by this reranker.
    """

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Return `results` unchanged; `top_n` is accepted for interface parity only."""
        return results
