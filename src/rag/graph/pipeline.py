"""`GraphRAGPipeline`: adds graph evidence to `RetrievalPipeline.answer()`.

`config.graph.fusion.strategy` selects the mode. With `"vector_only"` (the
default), `answer()` delegates unchanged to `RetrievalPipeline.answer()`, so
enabling the graph alone does not change answers. The other modes are
`graph_only`, `vector_plus_graph`, and `graph_seeded_vector`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rag.audit import log_audit_event
from rag.config import AppConfig
from rag.graph.client import GraphUnavailableError, Neo4jClient
from rag.graph.fusion import graph_facts_to_search_results
from rag.graph.retrieval import GraphRetriever
from rag.observability import metrics as observability_metrics
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.field_policy import sanitize_redaction_markers_in_answer
from rag.retrieval.injection_detection import detect_injection
from rag.retrieval.pipeline import RetrievalPipeline, source_dict
from rag.schemas import SearchResult

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Adds graph-based evidence to `RetrievalPipeline.answer()`, per `config.graph.fusion`."""

    def __init__(
        self,
        config: AppConfig,
        retrieval_pipeline: RetrievalPipeline,
        graph_client: Neo4jClient | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        """Wire up from config, the existing retrieval pipeline, and an optional graph client.

        Parameters
        ----------
        config : AppConfig
            Application configuration; `config.graph` supplies every
            fusion/traversal bound.
        retrieval_pipeline : RetrievalPipeline
            The existing vector/hybrid pipeline, always
            reused, never reimplemented.
        graph_client : Neo4jClient | None, optional
            Built from `config` if `config.graph.enabled` and omitted;
            left `None` (and every graph-touching code path skipped) when
            `config.graph.enabled=False`, so no driver is ever constructed
            in the default configuration.
        graph_retriever : GraphRetriever | None, optional
            Built from `config`/`graph_client` if omitted.
        """
        self._config = config
        self._retrieval_pipeline = retrieval_pipeline
        self._graph_enabled = config.graph.enabled
        self._client = graph_client
        self._retriever = graph_retriever
        if self._graph_enabled and self._client is None:
            self._client = Neo4jClient(config)
        if self._graph_enabled and self._retriever is None and self._client is not None:
            self._retriever = GraphRetriever(config, self._client, llm=retrieval_pipeline.llm)

    def close(self) -> None:
        """Close the underlying Neo4j client, if one was constructed."""
        if self._client is not None:
            self._client.close()

    def health_check(self) -> bool:
        """Return whether Neo4j is reachable, or `True` when graph is disabled.

        Used by `GET /health`. Neo4j is a soft dependency: an outage degrades
        the health report but does not stop the API from starting.
        """
        if not self._graph_enabled or self._client is None:
            return True
        return self._client.health_check()

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        candidate_k: int | None = None,
        auth: AuthorizationContext | None = None,
        strategy_override: str | None = None,
    ) -> dict[str, Any]:
        """Answer `query`, combining vector and graph evidence per the configured strategy.

        Returns the same dict shape as `RetrievalPipeline.answer()`, so callers
        can use either one.

        Parameters
        ----------
        query : str
            The user's query text.
        filters : dict[str, Any] | None, optional
            Exact-match metadata filters, passed through to vector
            retrieval and used to scope graph retrieval's `dataset_id`.
        candidate_k : int | None, optional
            Vector retrieval candidate depth override.
        auth : AuthorizationContext | None, optional
            Caller identity/date context, applied identically to both
            vector and graph retrieval.
        strategy_override : str | None, optional
            Replaces `config.graph.fusion.strategy` for this call only. Used by
            `POST /graph/query` and the graph evaluation. Ignored when
            `graph.enabled=False`.

        Returns
        -------
        dict[str, Any]
            Same shape as `RetrievalPipeline.answer()`.
        """
        requested_strategy = strategy_override or self._config.graph.fusion.strategy
        strategy = requested_strategy if self._graph_enabled else "vector_only"
        observability_metrics.observe_graphrag_route(strategy)

        if requested_strategy != "vector_only" and not self._graph_enabled:
            # A caller explicitly asked for graph evidence but
            # graph.enabled=False, so no graph retrieval can run at all.
            # Silently answering vector-only and letting the caller
            # believe the requested strategy ran is exactly the
            # mislabeling `graph_strategy_executed` exists to prevent;
            # `rag.eval.run_graph_eval.compare_strategies` checks this
            # same prerequisite up front and raises instead of reaching
            # this point for a scripted comparison. A direct, non-eval
            # caller (e.g. `POST /graph/query`) still gets a usable
            # vector-only answer rather than a hard failure, just
            # reported honestly.
            return self._vector_only_answer(
                query, filters, candidate_k, auth, executed_strategy="vector_only_graph_disabled"
            )

        if strategy == "vector_only" or self._retriever is None:
            return self._vector_only_answer(
                query,
                filters,
                candidate_k,
                auth,
                executed_strategy=(
                    "vector_only" if strategy == "vector_only" else "vector_only_graph_unavailable"
                ),
            )

        dataset_id = (filters or {}).get("dataset_id")
        if not dataset_id:
            # Graph retrieval is always scoped to one dataset. Without one, fall
            # back to vector-only rather than raising.
            return self._vector_only_answer(
                query, filters, candidate_k, auth, executed_strategy="vector_only_no_dataset_id"
            )

        if strategy == "graph_seeded_vector":
            return self._answer_graph_seeded_vector(query, filters, dataset_id, candidate_k, auth)
        if strategy == "graph_only":
            return self._answer_graph_only(query, dataset_id, auth)
        return self._answer_vector_plus_graph(query, filters, dataset_id, candidate_k, auth)

    def _vector_only_answer(
        self,
        query: str,
        filters: dict[str, Any] | None,
        candidate_k: int | None,
        auth: AuthorizationContext | None,
        *,
        executed_strategy: str,
    ) -> dict[str, Any]:
        """Delegate to `RetrievalPipeline.answer()`, tagging the result with what actually ran.

        Every plain-vector fallback path in this class goes through here
        so `graph_strategy_executed` is never missing from a response,
        whether the caller asked for `"vector_only"` explicitly or
        another strategy degraded into it.
        """
        result = self._retrieval_pipeline.answer(
            query, filters=filters, candidate_k=candidate_k, auth=auth
        )
        result["graph_strategy_executed"] = executed_strategy
        return result

    def _graph_search_results(
        self, query: str, dataset_id: str, auth: AuthorizationContext | None
    ) -> list[SearchResult]:
        """Run graph retrieval and return bounded, sanitized `SearchResult`s.

        Returns an empty list on graph failure. A degraded result is treated
        the same as an empty one.
        """
        assert self._retriever is not None and self._client is not None
        result = self._retriever.search(query, dataset_id=dataset_id, auth=auth)
        if result.degraded:
            observability_metrics.observe_graph_retrieval_failure("neo4j_unavailable")
        facts = result.facts[: self._config.graph.fusion.max_graph_evidence_items]
        try:
            search_results = graph_facts_to_search_results(facts, self._client, dataset_id)
        except GraphUnavailableError:
            logger.warning("Failed to resolve graph fact source paths", exc_info=True)
            return []
        # Same redaction and injection-flagging pass as every other evidence source.
        return self._retrieval_pipeline.sanitize_evidence(search_results, auth)

    def _generate_from_results(
        self,
        query: str,
        results: list[SearchResult],
        t0: float,
        retrieval_ms: float,
        auth: AuthorizationContext | None = None,
        executed_strategy: str = "vector_only",
    ) -> dict[str, Any]:
        """Render context from `results` and generate one answer, in `RetrievalPipeline.answer()`'s shape.

        Context is built via `RetrievalPipeline.build_context_text`
        (the same Context Engineering seam `answer()` and the agent's
        `_synthesize` node share), so graph evidence gets the same
        token-budget-aware treatment as vector evidence rather than a
        third, independent `build_context` call.
        """  # noqa: E501
        t1 = time.perf_counter()
        context, context_engineering_info, packed_results = (
            self._retrieval_pipeline.build_context_text(
                results, query, tenant_id=auth.tenant_id if auth else None
            )
        )
        system, user = self._retrieval_pipeline.prompt_template.render(context=context, query=query)
        answer_text = self._retrieval_pipeline.llm.generate(system, user)
        answer_text = sanitize_redaction_markers_in_answer(answer_text)
        t2 = time.perf_counter()
        generation_ms = (t2 - t1) * 1000
        graph_evidence_count = sum(1 for r in results if r.origin == "graph")
        observability_metrics.observe_graph_evidence_in_context(graph_evidence_count)
        return {
            "answer": answer_text,
            "prompt_tokens": getattr(self._retrieval_pipeline.llm, "last_prompt_tokens", None),
            "completion_tokens": getattr(
                self._retrieval_pipeline.llm, "last_completion_tokens", None
            ),
            "query_injection_suspected": detect_injection(query),
            # See RetrievalPipeline.answer()'s identical fix: sources
            # must reflect what the prompt actually contained
            # (EVAL-014), not pre-packing `results`.
            "sources": [source_dict(r) for r in packed_results],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "latency_breakdown_ms": {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms},
            "total_ms": (t2 - t0) * 1000,
            "context_engineering": context_engineering_info,
            "graph_strategy_executed": executed_strategy,
        }

    def _answer_graph_only(
        self, query: str, dataset_id: str, auth: AuthorizationContext | None
    ) -> dict[str, Any]:
        """`"graph_only"`: answer from graph evidence alone, no vector retrieval at all."""
        t0 = time.perf_counter()
        results = self._graph_search_results(query, dataset_id, auth)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return self._generate_from_results(
            query, results, t0, retrieval_ms, auth, executed_strategy="graph_only"
        )

    def _answer_vector_plus_graph(
        self,
        query: str,
        filters: dict[str, Any] | None,
        dataset_id: str,
        candidate_k: int | None,
        auth: AuthorizationContext | None,
    ) -> dict[str, Any]:
        """`"vector_plus_graph"`: run both, appending graph evidence after vector results."""
        t0 = time.perf_counter()
        vector_results = self._retrieval_pipeline.retrieve(
            query, filters=filters, candidate_k=candidate_k, auth=auth
        )
        graph_results = self._graph_search_results(query, dataset_id, auth)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        # Graph facts are appended after ranked vector results, never interleaved.
        combined = vector_results + graph_results
        return self._generate_from_results(
            query, combined, t0, retrieval_ms, auth, executed_strategy="vector_plus_graph"
        )

    def _answer_graph_seeded_vector(
        self,
        query: str,
        filters: dict[str, Any] | None,
        dataset_id: str,
        candidate_k: int | None,
        auth: AuthorizationContext | None,
    ) -> dict[str, Any]:
        """`"graph_seeded_vector"`: boost vector results from documents linked to query entities.

        Falls back to plain vector retrieval when no entity resolves or no
        linked documents are found. Seeding reorders results and never filters them.
        """
        assert self._retriever is not None and self._client is not None
        t0 = time.perf_counter()
        entities = self._retriever.resolve_entities(
            query,
            dataset_id=dataset_id,
            auth=auth,
            limit=self._config.graph.fusion.graph_seed_top_k,
        )
        if not entities:
            return self._vector_only_answer(
                query,
                filters,
                candidate_k,
                auth,
                executed_strategy="graph_seeded_vector_no_entities_resolved",
            )
        document_ids: set[str] = set()
        for entity in entities:
            neighbors = self._client.run_read(
                "MATCH (e {canonical_id: $id, dataset_id: $dataset_id})-[r]-(d:Document) "
                "RETURN DISTINCT d.canonical_id AS document_id LIMIT 20",
                {"id": entity.canonical_id, "dataset_id": dataset_id},
            )
            document_ids.update(row["document_id"] for row in neighbors)
        log_audit_event(
            "graph_seeded_vector_retrieval",
            dataset_id=dataset_id,
            resolved_entity_count=len(entities),
            seeded_document_count=len(document_ids),
        )
        if not document_ids:
            return self._vector_only_answer(
                query,
                filters,
                candidate_k,
                auth,
                executed_strategy="graph_seeded_vector_no_linked_documents",
            )
        # A seed narrows candidate_k's own pool via a broader fetch plus a
        # post-hoc boost, not a hard filter (ALLOWED_FILTER_FIELDS has no
        # "IN a set of ids" operator for document_id), so retrieve normally,
        # then stable-sort seeded documents' results ahead of the rest.
        results = self._retrieval_pipeline.retrieve(
            query, filters=filters, candidate_k=candidate_k, auth=auth
        )
        results = sorted(results, key=lambda r: r.chunk.metadata.document_id not in document_ids)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return self._generate_from_results(
            query, results, t0, retrieval_ms, auth, executed_strategy="graph_seeded_vector"
        )
