"""`Reranker` backed by the Cohere rerank API."""

from __future__ import annotations

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class CohereReranker(Reranker):
    """Optional provider, active only if reranker.provider is 'cohere' and its API key is set.

    Requires the 'cohere' extra: pip install .[cohere]
    """

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0") -> None:
        """Construct the Cohere client.

        Parameters
        ----------
        api_key : str
            Cohere API key.
        model_name : str, optional
            Cohere rerank model id, by default ``"rerank-english-v3.0"``.

        Raises
        ------
        RuntimeError
            If the optional ``cohere`` package isn't installed.
        """
        try:
            import cohere
        except ImportError as exc:
            raise RuntimeError(
                "reranker.provider is 'cohere' but the 'cohere' package isn't installed. "
                "Install it with: pip install .[cohere]"
            ) from exc
        self._client = cohere.Client(api_key)
        self._model_name = model_name

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Rescore `results` against `query` via the Cohere rerank API.

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
        response = self._client.rerank(
            query=query,
            documents=[r.chunk.content for r in results],
            top_n=min(top_n, len(results)),
            model=self._model_name,
        )
        return [
            results[item.index].model_copy(update={"score": float(item.relevance_score)})
            for item in response.results
        ]
