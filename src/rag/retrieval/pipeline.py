"""Query pipeline: embed query -> vectorstore.search(filters) -> rerank -> generate."""

from __future__ import annotations

from typing import Any

from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_llm, build_reranker, build_vectorstore
from rag.generation.base import LLM
from rag.rerankers.base import Reranker
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore

_PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {query}

Answer:"""


class RetrievalPipeline:
    def __init__(
        self,
        config: AppConfig,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore or build_vectorstore(config)
        self._embedder = embedder or build_embedder(config)
        self._reranker = reranker or build_reranker(config)
        self._llm = llm or build_llm(config)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        query_embedding = self._embedder.embed_query(query)
        results = self._vectorstore.search(
            query_embedding, top_k=top_k or self._config.retrieval.top_k, filters=filters
        )
        return self._reranker.rerank(query, results, top_n=self._config.retrieval.rerank_top_n)

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        results = self.retrieve(query, filters=filters, top_k=top_k)
        context = "\n\n---\n\n".join(r.chunk.content for r in results)
        prompt = _PROMPT_TEMPLATE.format(context=context, query=query)
        answer_text = self._llm.generate(prompt)
        return {
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": r.chunk.metadata.chunk_id,
                    "document_id": r.chunk.metadata.document_id,
                    "source": r.chunk.metadata.source,
                    "score": r.score,
                }
                for r in results
            ],
        }
