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
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.schemas import Chunk, SearchResult
from rag.vectorstore.base import VectorStore


def _source_label(result: SearchResult) -> str:
    """Build one result's "[Source N: ...]" provenance label (without the leading "[Source N: ").

    Includes section, content type, and -- for an image chunk -- whether
    its text is a caption/alt-text or a vision-generated description, so
    the prompt never gives the model grounds to imply it visually
    inspected an image when only caption/alt-text was available (see
    prompt v2's "preserve exact ... verbatim" / no-unsupported-claims
    rules). Relationship-expanded results are labeled "related context" so
    they're identifiable as such, not indistinguishable from a directly
    retrieved match.
    """
    meta = result.chunk.metadata
    parts = [meta.source]
    if meta.section_path:
        parts.append(meta.section_path)
    content_type = meta.content_type or "prose"
    parts.append(content_type)
    if content_type == "image":
        parts.append(
            "vision-generated description" if meta.vision_generated else "caption/alt-text only"
        )
    if result.origin == "expanded":
        parts.append("related context")
    return " | ".join(parts)


def _build_context(results: list[SearchResult]) -> str:
    """Join reranked chunks into a "[Source N: ...]"-labeled context block."""
    labeled = [
        f"[Source {i}: {_source_label(r)}]\n{r.chunk.content}"
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
        """Retrieve candidates (dense-only or hybrid, per `config.retrieval.provider`), then rerank.

        When `config.retrieval.provider == "hybrid"`, dense (embedding)
        search and keyword (BM25) search are each run at `top_k`, then
        fused via Reciprocal Rank Fusion
        (`config.retrieval.hybrid.rrf_k`) before reranking. When
        `"dense"` (the default), behavior is unchanged from plain vector
        search.

        `rerank_top_n` overrides config's default truncation — needed by
        callers (e.g. eval) that want more than the production-default
        number of results back, such as computing Recall@10 when the
        configured reranker normally truncates to top 3.

        When `config.retrieval.relationship_expansion.enabled`, parent/
        neighbor context is appended after reranking (see
        `_expand_with_relationships`) -- disabled by default, a no-op when
        left that way.

        Parameters
        ----------
        query : str
            The user's query text.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters passed through to
            `VectorStore.search`/`search_keyword`.
        top_k : int | None, optional
            Number of results to fetch from the vector store (from each
            branch, in hybrid mode); defaults to `config.retrieval.top_k`.
        rerank_top_n : int | None, optional
            Number of results to keep after reranking; defaults to
            `config.retrieval.rerank_top_n`.

        Returns
        -------
        list[SearchResult]
            Reranked results, best-first, plus any relationship-expanded
            results appended after them (see `origin`/`expanded_from` on
            `SearchResult`).
        """
        query_embedding = self._embedder.embed_query(query)
        fetch_k = top_k or self._config.retrieval.top_k
        if self._config.retrieval.provider == "hybrid":
            dense_results = self._vectorstore.search(
                query_embedding, top_k=fetch_k, filters=filters
            )
            keyword_results = self._vectorstore.search_keyword(
                query, top_k=fetch_k, filters=filters
            )
            candidates = reciprocal_rank_fusion(
                [dense_results, keyword_results], k=self._config.retrieval.hybrid.rrf_k
            )
        else:
            candidates = self._vectorstore.search(query_embedding, top_k=fetch_k, filters=filters)
        n = rerank_top_n if rerank_top_n is not None else self._config.retrieval.rerank_top_n
        reranked = self._reranker.rerank(query, candidates, top_n=n)
        if self._config.retrieval.relationship_expansion.enabled:
            return self._expand_with_relationships(reranked)
        return reranked

    def _expand_with_relationships(self, results: list[SearchResult]) -> list[SearchResult]:
        """Append parent/neighbor context for each result, per `retrieval.relationship_expansion`.

        Ranking and expansion stay separate: expanded chunks are appended
        after the ranked list (`origin="expanded"`, `expanded_from=<id>`),
        never interleaved into it or given a freshly computed score --
        `.score` is inherited from the originating result, so `origin` (not
        `.score`) is what callers should check to tell directly-retrieved
        context from expanded context apart. A `parent_chunk_id`/section
        lookup can never cross a `dataset_id` boundary because it's always
        scoped to one already-`dataset_id`-filtered result's own
        `document_id` (see `VectorStore.get_chunks_by_ids`/
        `get_chunks_by_section`). Deduplicates against chunks already
        present in `results` and across expansions of different results,
        and caps additions at `max_related_elements` per originating result.

        Parameters
        ----------
        results : list[SearchResult]
            The reranked, directly-retrieved results to expand.

        Returns
        -------
        list[SearchResult]
            `results` followed by any expanded `SearchResult`s.
        """
        cfg = self._config.retrieval.relationship_expansion
        if not results:
            return results

        present_ids = {r.chunk.metadata.chunk_id for r in results}

        parents_by_id: dict[str, Chunk] = {}
        if cfg.include_parent:
            needed = {
                r.chunk.metadata.parent_chunk_id
                for r in results
                if r.chunk.metadata.parent_chunk_id
                and r.chunk.metadata.parent_chunk_id not in present_ids
            }
            if needed:
                for chunk in self._vectorstore.get_chunks_by_ids(list(needed)):
                    parents_by_id[chunk.metadata.chunk_id] = chunk

        section_cache: dict[tuple[str, str | None], list[Chunk]] = {}
        added_ids: set[str] = set()
        expanded: list[SearchResult] = []

        for result in results:
            meta = result.chunk.metadata
            related: list[Chunk] = []

            if cfg.include_parent and meta.parent_chunk_id:
                parent = parents_by_id.get(meta.parent_chunk_id)
                if parent is not None:
                    related.append(parent)

            if cfg.include_neighbors:
                section_key = (meta.document_id, meta.section_path)
                if section_key not in section_cache:
                    section_cache[section_key] = self._vectorstore.get_chunks_by_section(
                        meta.document_id, meta.section_path
                    )
                siblings = section_cache[section_key]
                position = next(
                    (i for i, c in enumerate(siblings) if c.metadata.chunk_id == meta.chunk_id),
                    None,
                )
                if position is not None:
                    if position > 0:
                        related.append(siblings[position - 1])
                    if position + 1 < len(siblings):
                        related.append(siblings[position + 1])

            added_for_result = 0
            for chunk in related:
                if added_for_result >= cfg.max_related_elements:
                    break
                chunk_id = chunk.metadata.chunk_id
                if chunk_id in present_ids or chunk_id in added_ids:
                    continue
                added_ids.add(chunk_id)
                added_for_result += 1
                expanded.append(
                    SearchResult(
                        chunk=chunk,
                        score=result.score,
                        origin="expanded",
                        expanded_from=meta.chunk_id,
                    )
                )

        return results + expanded

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
            Each `sources` entry includes the chunk's raw `content`
            alongside its metadata (including `content_type`,
            `attachment_name`/`source_anchor` for image hits, and
            `origin`/`expanded_from` for relationship-expansion
            provenance), so callers (e.g. RAGAS scoring, eval's
            reference-context/image-hit metrics) can reuse this retrieval
            without a redundant call.
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
                    "content": r.chunk.content,
                    "content_type": r.chunk.metadata.content_type,
                    "section_path": r.chunk.metadata.section_path,
                    "attachment_name": r.chunk.metadata.attachment_name,
                    "source_anchor": r.chunk.metadata.source_anchor,
                    "vision_generated": r.chunk.metadata.vision_generated,
                    "origin": r.origin,
                    "expanded_from": r.expanded_from,
                }
                for r in results
            ],
            "retrieval_ms": (t1 - t0) * 1000,
            "generation_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        }
