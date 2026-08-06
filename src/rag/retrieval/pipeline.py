"""Query pipeline: embed query -> vectorstore.search(filters) -> rerank -> generate."""

from __future__ import annotations

import time
from typing import Any

from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.factory import build_embedder, build_llm, build_reranker, build_vectorstore
from rag.generation.base import LLM
from rag.prompts.loader import PromptTemplate, load_prompt_template_from_config
from rag.rerankers.base import Reranker
from rag.schemas import SearchResult
from rag.vectorstore.base import VectorStore


def _build_context(results: list[SearchResult]) -> str:
    """Join reranked chunks into a "[Source N: ...]"-labeled context block."""
    labeled = [
        f"[Source {i}: {r.chunk.metadata.source}]\n{r.chunk.content}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n\n---\n\n".join(labeled)


class RetrievalPipeline:
    """Answers a query via embed -> vectorstore.search -> rerank -> generate."""

    def __init__(
        self,
        config: AppConfig,
        vectorstore: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
        prompt_template: PromptTemplate | None = None,
    ) -> None:
        """Wire up the pipeline's stages from config (or injected instances).

        Parameters
        ----------
        config : AppConfig
            Application configuration.
        vectorstore : VectorStore | None, optional
            Vector store to search; built from `config` if omitted.
        embedder : Embedder | None, optional
            Embedder used for the query vector; built from `config` if omitted.
        reranker : Reranker | None, optional
            Reranker applied to search results; built from `config` if omitted.
        llm : LLM | None, optional
            LLM used for generation; built from `config` if omitted.
        prompt_template : PromptTemplate | None, optional
            Prompt template used for generation; loaded from
            `config.generation.prompt` if omitted.
        """
        self._config = config
        self._vectorstore = vectorstore or build_vectorstore(config)
        self._embedder = embedder or build_embedder(config)
        self._reranker = reranker or build_reranker(config)
        self._llm = llm or build_llm(config)
        self._prompt_template = prompt_template or load_prompt_template_from_config(config)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> list[SearchResult]:
        """Embed `query`, search the vector store, and rerank the results.

        `rerank_top_n` overrides config's default truncation — needed by
        callers (e.g. eval) that want more than the production-default
        number of results back, such as computing Recall@10 when the
        configured reranker normally truncates to top 3.

        Parameters
        ----------
        query : str
            The user's query text.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters passed through to
            `VectorStore.search`.
        top_k : int | None, optional
            Number of results to fetch from the vector store; defaults to
            `config.retrieval.top_k`.
        rerank_top_n : int | None, optional
            Number of results to keep after reranking; defaults to
            `config.retrieval.rerank_top_n`.

        Returns
        -------
        list[SearchResult]
            Reranked results, best-first.
        """
        query_embedding = self._embedder.embed_query(query)
        results = self._vectorstore.search(
            query_embedding, top_k=top_k or self._config.retrieval.top_k, filters=filters
        )
        n = rerank_top_n if rerank_top_n is not None else self._config.retrieval.rerank_top_n
        return self._reranker.rerank(query, results, top_n=n)

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve context for `query` and generate an answer from it.

        Always reflects production config (no `rerank_top_n` override —
        see `retrieve`).

        Parameters
        ----------
        query : str
            The user's query text.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters passed through to `retrieve`.
        top_k : int | None, optional
            Number of results to fetch from the vector store; defaults to
            `config.retrieval.top_k`.

        Returns
        -------
        dict[str, Any]
            ``{"answer", "sources", "retrieval_ms", "generation_ms", "total_ms"}``.
        """
        t0 = time.perf_counter()
        results = self.retrieve(query, filters=filters, top_k=top_k)
        t1 = time.perf_counter()
        context = _build_context(results)
        system, user = self._prompt_template.render(context=context, query=query)
        prompt = f"{system}\n\n{user}" if system else user
        answer_text = self._llm.generate(prompt)
        t2 = time.perf_counter()
        return {
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": r.chunk.metadata.chunk_id,
                    "document_id": r.chunk.metadata.document_id,
                    "source": r.chunk.metadata.source,
                    "category": r.chunk.metadata.category,
                    "score": r.score,
                }
                for r in results
            ],
            "retrieval_ms": (t1 - t0) * 1000,
            "generation_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        }
