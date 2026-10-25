"""Compare vector-only and GraphRAG fusion strategies on the graph gold set.

Runs each strategy over `data/eval/graph_gold.jsonl` by overriding
`GraphRAGPipeline.answer()`'s strategy per call. It does not reuse
`eval/run_eval.py`, which covers the full multimodal and safety gold schema.

Usage:
    python -m rag.eval.run_graph_eval --gold data/eval/graph_gold.jsonl
        --dataset-id techfusion_graph_demo
        --strategies vector_only,graph_only,vector_plus_graph,graph_seeded_vector
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from typing import Any

from rag.api.deps import get_config
from rag.config import AppConfig
from rag.eval.answer_quality import KeywordOverlapScorer
from rag.eval.gold_schema import GoldExample, load_gold_jsonl, source_matches_relevant
from rag.factory import build_graph_client
from rag.graph.pipeline import GraphRAGPipeline
from rag.graph.retrieval import GraphRetriever
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline


def _build_auth(example: GoldExample) -> AuthorizationContext | None:
    """Build an `AuthorizationContext` from a gold row's simulated caller identity."""
    if not (example.user_tenant or example.user_roles):
        return None
    query_as_of = date.fromisoformat(example.query_as_of) if example.query_as_of else None
    return AuthorizationContext(
        tenant_id=example.user_tenant, roles=example.user_roles, as_of=query_as_of
    )


def _entity_mention_hit(answer_text: str, expected_entities: list[str]) -> bool | None:
    """Whether the answer mentions at least one of the expected graph entities.

    `None` when the gold row names no expected entities (nothing to check).
    """
    if not expected_entities:
        return None
    lowered = answer_text.lower()
    return any(name.lower() in lowered for name in expected_entities)


def _citation_hit(sources: list[dict[str, Any]], relevant_documents: list[str]) -> bool | None:
    """Whether any cited source matches a gold `relevant_documents` entry, by path suffix."""
    if not relevant_documents:
        return None
    retrieved_sources = [s["source"] for s in sources]
    return any(
        source_matches_relevant(s, rd) for s in retrieved_sources for rd in relevant_documents
    )


def _unanswerable_correctly_handled(answer_text: str) -> bool:
    """Return whether an unanswerable question got a refusal-style answer.

    Literal phrase matching. This is a triage aid, not a correctness check.
    """
    lowered = answer_text.lower()
    return any(
        phrase in lowered
        for phrase in ("don't have", "cannot find", "no information", "not able to", "unable to")
    )


def evaluate_strategy(
    gold: list[GoldExample],
    pipeline: GraphRAGPipeline,
    strategy: str,
    dataset_id: str,
) -> dict[str, Any]:
    """Run every gold example through one fusion strategy and aggregate metrics.

    Parameters
    ----------
    gold : list[GoldExample]
        The graph-specific gold set.
    pipeline : GraphRAGPipeline
        The pipeline to answer through.
    strategy : str
        One of `"vector_only"`/`"graph_only"`/`"vector_plus_graph"`/
        `"graph_seeded_vector"`, passed as `answer()`'s `strategy_override`.
    dataset_id : str
        Dataset namespace every gold row is answered within.

    Returns
    -------
    dict[str, Any]
        Aggregate metrics plus a `per_example` list, matching this
        project's established `{"count", "rate", ...}` metric shape where
        applicable.
    """
    scorer = KeywordOverlapScorer()
    per_example: list[dict[str, Any]] = []
    total_latency_ms = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    degraded_count = 0

    for example in gold:
        auth = _build_auth(example)
        t0 = time.perf_counter()
        result = pipeline.answer(
            example.question,
            filters={"dataset_id": dataset_id},
            auth=auth,
            strategy_override=strategy,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency_ms += latency_ms
        total_prompt_tokens += result.get("prompt_tokens") or 0
        total_completion_tokens += result.get("completion_tokens") or 0
        # EVAL-013: `result["graph_strategy_executed"]` is what the
        # pipeline actually ran, independent of the `strategy` label this
        # loop requested; GraphRAGPipeline.answer() can silently degrade
        # to plain vector retrieval per-query (no entities resolved, no
        # dataset_id, graph outage). A mismatch here means this row's
        # metrics reflect a different mechanism than its own `strategy`
        # column claims.
        executed = result.get("graph_strategy_executed", strategy)
        if executed != strategy:
            degraded_count += 1

        answer_quality = scorer.score(example.question, result["answer"], example.expected_answer)
        entity_hit = _entity_mention_hit(result["answer"], example.expected_graph_entities)
        citation_hit = _citation_hit(result["sources"], example.relevant_documents)
        graph_evidence_used = any(s.get("origin") == "graph" for s in result["sources"])
        unauthorized_leak = None
        if example.forbidden_documents:
            unauthorized_leak = any(
                source_matches_relevant(s["source"], fd)
                for s in result["sources"]
                for fd in example.forbidden_documents
            )
        unanswerable_handled = (
            _unanswerable_correctly_handled(result["answer"])
            if (example.unanswerable or example.graph_unanswerable)
            else None
        )

        per_example.append(
            {
                "question": example.question,
                "graph_category": example.graph_category,
                "graph_traversal_depth": example.graph_traversal_depth,
                "answer_quality": answer_quality,
                "entity_hit": entity_hit,
                "citation_hit": citation_hit,
                "graph_evidence_used": graph_evidence_used,
                "unauthorized_leak": unauthorized_leak,
                "unanswerable_handled": unanswerable_handled,
                "latency_ms": latency_ms,
                "answer": result["answer"],
                "sources": [s["source"] for s in result["sources"]],
                "graph_strategy_executed": executed,
                "degraded": executed != strategy,
            }
        )

    def _rate(key: str) -> dict[str, Any]:
        values = [row[key] for row in per_example if row[key] is not None]
        if not values:
            return {"count": 0, "rate": None}
        return {"count": len(values), "rate": sum(1 for v in values if v) / len(values)}

    return {
        "strategy": strategy,
        "example_count": len(gold),
        "degraded_run_count": degraded_count,
        "degraded_run_rate": degraded_count / len(gold) if gold else 0.0,
        "mean_answer_quality": (
            sum(row["answer_quality"] for row in per_example) / len(per_example)
            if per_example
            else 0.0
        ),
        "entity_hit_rate": _rate("entity_hit"),
        "citation_hit_rate": _rate("citation_hit"),
        "graph_evidence_usage_rate": {
            "count": len(per_example),
            "rate": (
                sum(1 for row in per_example if row["graph_evidence_used"]) / len(per_example)
                if per_example
                else 0.0
            ),
        },
        "unauthorized_leak_rate": _rate("unauthorized_leak"),
        "unanswerable_handled_rate": _rate("unanswerable_handled"),
        "mean_latency_ms": total_latency_ms / len(gold) if gold else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "per_example": per_example,
    }


def compare_strategies(
    gold_path: str, dataset_id: str, strategies: list[str], config: AppConfig | None = None
) -> dict[str, Any]:
    """Run every strategy in `strategies` over `gold_path` and return a combined comparison report.

    Parameters
    ----------
    gold_path : str
        Path to the graph gold JSONL file.
    dataset_id : str
        Dataset namespace.
    strategies : list[str]
        Fusion strategies to compare, e.g.
        `["vector_only", "graph_only", "vector_plus_graph", "graph_seeded_vector"]`.
    config : AppConfig | None, optional
        Application configuration; loaded via `get_config()` if omitted.
        Must have `graph.enabled=True` for any non-`"vector_only"`
        strategy to produce real graph evidence.

    Returns
    -------
    dict[str, Any]
        `{"dataset_id", "gold_count", "results": {strategy: evaluate_strategy(...)}}`.

    Raises
    ------
    ValueError
        If `strategies` names any non-`"vector_only"` mode while
        `config.graph.enabled` is `False`. Regression for EVAL-013:
        `GraphRAGPipeline.answer()` silently substitutes plain vector
        retrieval whenever graph is disabled, previously still labeled
        with whatever strategy was requested, so a comparison run under
        the shipped (graph-disabled) default could report four
        differently-named rows that all executed identically. Checked
        here, once, before any row runs, rather than only discoverable
        per-row via `evaluate_strategy`'s `degraded_run_rate`.
    """
    config = config or get_config()
    if not config.graph.enabled and any(s != "vector_only" for s in strategies):
        requested = [s for s in strategies if s != "vector_only"]
        raise ValueError(
            f"compare_strategies was asked to run {requested} but config.graph.enabled is "
            "False: every one of those strategies would silently execute plain vector "
            "retrieval instead, which would be reported under the wrong strategy label. "
            'Enable config.graph.enabled, or request only ["vector_only"].'
        )
    gold = load_gold_jsonl(gold_path)
    retrieval_pipeline = RetrievalPipeline(config)
    graph_client = build_graph_client(config)
    graph_retriever = (
        GraphRetriever(config, graph_client, llm=retrieval_pipeline.llm)
        if graph_client is not None
        else None
    )
    pipeline = GraphRAGPipeline(
        config, retrieval_pipeline, graph_client=graph_client, graph_retriever=graph_retriever
    )
    try:
        results = {
            strategy: evaluate_strategy(gold, pipeline, strategy, dataset_id)
            for strategy in strategies
        }
    finally:
        pipeline.close()
    return {"dataset_id": dataset_id, "gold_count": len(gold), "results": results}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to the graph gold JSONL file")
    parser.add_argument("--dataset-id", required=True, help="Dataset namespace to query within")
    parser.add_argument(
        "--strategies",
        default="vector_only,graph_only,vector_plus_graph,graph_seeded_vector",
        help="Comma-separated fusion strategies to compare",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument("--output", default=None, help="Write the full JSON report to this path")
    args = parser.parse_args()

    from rag.config import load_config

    config = load_config(args.config) if args.config else get_config()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    report = compare_strategies(args.gold, args.dataset_id, strategies, config=config)

    for strategy, metrics in report["results"].items():
        print(f"\n=== {strategy} ===")
        print(f"  mean_answer_quality: {metrics['mean_answer_quality']:.3f}")
        print(f"  entity_hit_rate: {metrics['entity_hit_rate']}")
        print(f"  citation_hit_rate: {metrics['citation_hit_rate']}")
        print(f"  graph_evidence_usage_rate: {metrics['graph_evidence_usage_rate']}")
        print(f"  unauthorized_leak_rate: {metrics['unauthorized_leak_rate']}")
        print(f"  unanswerable_handled_rate: {metrics['unanswerable_handled_rate']}")
        print(f"  mean_latency_ms: {metrics['mean_latency_ms']:.1f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
