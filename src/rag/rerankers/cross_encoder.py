"""`Reranker` backed by a local sentence-transformers CrossEncoder model."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class CrossEncoderReranker(Reranker):
    """Local reranker via sentence-transformers CrossEncoder.

    Adds CPU latency per query. Swap in via config when relevance
    matters more than speed.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"
    ) -> None:
        """Load the cross-encoder model once.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face model id, by default
            ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
        device : str, optional
            Torch device to run inference on, by default ``"cpu"``.
        """
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Rescore `results` against `query` and return the top-n by that score.

        Parameters
        ----------
        query : str
            The original user query.
        results : list[SearchResult]
            Vector-search results to rerank.
        top_n : int
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Up to `top_n` results, rescored and sorted best-first.
        """
        if not results:
            return results
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)
        rescored = [
            r.model_copy(update={"score": float(s)}) for r, s in zip(results, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_n]
