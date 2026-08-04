from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class CrossEncoderReranker(Reranker):
    """Local reranker via sentence-transformers CrossEncoder. Adds CPU
    latency per query — swap in via config when relevance matters more
    than speed."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"
    ) -> None:
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)
        rescored = [
            r.model_copy(update={"score": float(s)}) for r, s in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_n]
