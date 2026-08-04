from __future__ import annotations

from rag.rerankers.base import Reranker
from rag.schemas import SearchResult


class CohereReranker(Reranker):
    """Optional provider — only active if reranker.provider is set to
    'cohere' AND its API key env var is set. Requires the 'cohere' extra:
    pip install .[cohere]"""

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0") -> None:
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
