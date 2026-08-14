"""Measure what dense retrieval, BM25, and RRF fusion each contribute, independently.

An observability/evaluation milestone, not a retrieval-tuning one: nothing
here changes `config/default.yaml`, `RetrievalPipeline.retrieve()`/
`answer()`, RRF's `k`, the reranker, embeddings, chunking, the prompt
version, or the generation model. `RetrievalPipeline.retrieve_attribution`
(see that method's docstring) is the one small, additive pipeline change
this milestone needed -- it fetches dense and BM25 rankings independently
and fuses them via the same `reciprocal_rank_fusion` production uses,
without ever calling the reranker or relationship expansion.

Fully deterministic and local: real Postgres/pgvector retrieval, no LLM
generation, no hosted judge call. Reuses `eval/metrics.py` (Recall/Hit
Rate/MRR) and `eval/gold_schema.py`'s `source_matches_relevant`/
`reference_context_bucket` rather than inventing parallel definitions.

Ranks, not raw scores, are the primary comparison throughout: a dense
cosine-similarity score and a BM25 score are not on the same scale (see
`retrieval/fusion.py`'s own docstring for why RRF itself is rank-based),
so this module never compares `SearchResult.score` across retrievers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import AppConfig, load_config
from rag.eval.gold_schema import (
    GoldExample,
    load_gold_jsonl,
    reference_context_bucket,
    source_matches_relevant,
)
from rag.eval.metrics import mean_hit_rate_at_k, mean_recall_at_k, mean_reciprocal_rank
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import SearchResult

# Broad, fixed cutoff fetched per retriever -- matches run_eval.py's
# RETRIEVAL_K, so dense/BM25/hybrid attribution and the production
# Recall@10 measurement are all "at the target cutoff" in the same sense.
RETRIEVAL_K = 10
RECALL_CUTOFFS = (5, 10)
RETRIEVERS = ("dense", "bm25", "hybrid")

_UNCATEGORIZED = "uncategorized"


def _sources(results: list[SearchResult]) -> list[str]:
    """Extract each result's chunk `source` path, preserving rank order."""
    return [r.chunk.metadata.source for r in results]


def _contents(results: list[SearchResult]) -> list[str]:
    """Extract each result's chunk text content, preserving rank order."""
    return [r.chunk.content for r in results]


def first_relevant_rank(sources: list[str], relevant_documents: list[str]) -> int | None:
    """1-based rank of the first source matching any relevant document, or None.

    Parameters
    ----------
    sources : list[str]
        A ranking's retrieved `source` paths, best-first.
    relevant_documents : list[str]
        The gold example's `relevant_documents`.

    Returns
    -------
    int | None
        The rank (1 = first result) of the first match, or `None` if no
        `sources` entry matches any `relevant_documents` entry.
    """
    for rank, source in enumerate(sources, start=1):
        if any(source_matches_relevant(source, rel) for rel in relevant_documents):
            return rank
    return None


def classify_contribution(dense_hit: bool, bm25_hit: bool) -> str:
    """Classify a question's dense-vs-BM25 success pattern.

    Parameters
    ----------
    dense_hit : bool
        Whether dense retrieval alone found a relevant document within
        `RETRIEVAL_K`.
    bm25_hit : bool
        Whether BM25 retrieval alone found a relevant document within
        `RETRIEVAL_K`.

    Returns
    -------
    str
        One of ``"both_success"``, ``"dense_only_success"``,
        ``"bm25_only_success"``, ``"neither_success"``.
    """
    if dense_hit and bm25_hit:
        return "both_success"
    if dense_hit:
        return "dense_only_success"
    if bm25_hit:
        return "bm25_only_success"
    return "neither_success"


def classify_rrf_impact(
    dense_rank: int | None, bm25_rank: int | None, fused_rank: int | None
) -> str:
    """Classify how RRF fusion changed the rank of the first relevant document.

    A deterministic decision tree over the three ranks (each the 1-based
    rank of the first relevant document within that ranking's
    `RETRIEVAL_K`-sized fetch, or `None` if not found there at all):

    - ``"not_applicable"``: neither dense nor BM25 found it -- there is
      nothing for fusion to have rescued, improved, or degraded.
    - ``"rescued"``: exactly one of dense/BM25 missed it entirely, but the
      fused ranking still found it -- fusion recovered evidence one
      retriever alone would have missed at this cutoff.
    - ``"still_missed"``: exactly one of dense/BM25 missed it, and fusion
      missed it too.
    - ``"improved"``: both dense and BM25 found it, and the fused rank is
      strictly better (numerically lower) than the better of the two.
    - ``"unchanged"``: both found it, and the fused rank equals the
      better of the two.
    - ``"degraded"``: both found it, but the fused rank is worse than the
      better of the two, or (structurally rare, since RRF's candidate set
      is the union of both inputs, but handled for completeness) fusion
      missed it even though both individual retrievers found it.

    Parameters
    ----------
    dense_rank : int | None
        First-relevant-document rank in the dense-only ranking.
    bm25_rank : int | None
        First-relevant-document rank in the BM25-only ranking.
    fused_rank : int | None
        First-relevant-document rank in the RRF-fused ranking.

    Returns
    -------
    str
        One of the six categories above.
    """
    if dense_rank is None and bm25_rank is None:
        return "not_applicable"
    if dense_rank is None or bm25_rank is None:
        return "rescued" if fused_rank is not None else "still_missed"
    best_single = min(dense_rank, bm25_rank)
    if fused_rank is None:
        return "degraded"
    if fused_rank < best_single:
        return "improved"
    if fused_rank == best_single:
        return "unchanged"
    return "degraded"


def _retriever_metrics(
    retrieved_sources: list[list[str]], all_relevant: list[list[str]]
) -> dict[str, float]:
    """Recall@5/@10, Hit Rate@5/@10, and MRR for one retriever's rankings."""
    metrics = {
        f"recall@{k}": mean_recall_at_k(retrieved_sources, all_relevant, k, source_matches_relevant)
        for k in RECALL_CUTOFFS
    }
    metrics.update(
        {
            f"hit_rate@{k}": mean_hit_rate_at_k(
                retrieved_sources, all_relevant, k, source_matches_relevant
            )
            for k in RECALL_CUTOFFS
        }
    )
    metrics["mrr"] = mean_reciprocal_rank(retrieved_sources, all_relevant, source_matches_relevant)
    return metrics


def _breakdown_by(
    examples: list[GoldExample],
    per_retriever_sources: dict[str, list[list[str]]],
    all_relevant: list[list[str]],
    key_fn: Callable[[GoldExample], str],
) -> dict[str, dict[str, Any]]:
    """Bucket examples by `key_fn`, reporting count + per-retriever Recall@5/@10/Hit Rate@5.

    Always reports `count` alongside every bucket's numbers, so a
    single-digit-sample bucket's percentage is never shown without the
    sample size behind it.
    """
    bucket_indices: dict[str, list[int]] = {}
    for i, example in enumerate(examples):
        bucket_indices.setdefault(key_fn(example), []).append(i)

    breakdown: dict[str, dict[str, Any]] = {}
    for bucket, indices in sorted(bucket_indices.items()):
        relevant = [all_relevant[i] for i in indices]
        by_retriever = {}
        for name in RETRIEVERS:
            retrieved = [per_retriever_sources[name][i] for i in indices]
            by_retriever[name] = {
                "recall@5": mean_recall_at_k(retrieved, relevant, 5, source_matches_relevant),
                "recall@10": mean_recall_at_k(retrieved, relevant, 10, source_matches_relevant),
                "hit_rate@5": mean_hit_rate_at_k(retrieved, relevant, 5, source_matches_relevant),
            }
        breakdown[bucket] = {"count": len(indices), "by_retriever": by_retriever}
    return breakdown


def evaluate_attribution(
    pipeline: RetrievalPipeline, examples: list[GoldExample], dataset_id: str
) -> dict[str, Any]:
    """Run dense/BM25/hybrid attribution over `examples` and score the results.

    Every retrieval call is restricted to `dataset_id` via a mandatory
    filter (same isolation guarantee as `run_eval.evaluate`). Uses
    `RetrievalPipeline.retrieve_attribution` exclusively -- never
    `retrieve()`/`answer()` -- so nothing here goes through the reranker
    or relationship expansion.

    Parameters
    ----------
    pipeline : RetrievalPipeline
        Pipeline to evaluate.
    examples : list[GoldExample]
        Gold questions to run.
    dataset_id : str
        Namespace to restrict every retrieval to.

    Returns
    -------
    dict[str, Any]
        Report with `metrics_by_retriever` (dense/bm25/hybrid Recall@5/@10,
        Hit Rate@5/@10, MRR, all computed over the identical gold records),
        `contribution_buckets`, `rrf_impact`, `reference_context_by_retriever`,
        `breakdowns` (by content_type/question_type/difficulty/
        relationship_aware), and `per_example` diagnostics.
    """
    dataset_filter = {"dataset_id": dataset_id}

    per_retriever_sources: dict[str, list[list[str]]] = {name: [] for name in RETRIEVERS}
    all_relevant: list[list[str]] = []
    contribution_counts: Counter[str] = Counter()
    rrf_impact_counts: Counter[str] = Counter()
    reference_context_buckets: dict[str, list[str]] = {name: [] for name in RETRIEVERS}
    per_example: list[dict[str, Any]] = []

    for example in examples:
        attribution = pipeline.retrieve_attribution(
            example.question, filters=dataset_filter, candidate_k=RETRIEVAL_K
        )
        results_by_retriever = {
            "dense": attribution.dense,
            "bm25": attribution.bm25,
            "hybrid": attribution.fused,
        }
        sources_by_retriever = {
            name: _sources(results) for name, results in results_by_retriever.items()
        }
        contents_by_retriever = {
            name: _contents(results) for name, results in results_by_retriever.items()
        }
        for name in RETRIEVERS:
            per_retriever_sources[name].append(sources_by_retriever[name])
        all_relevant.append(example.relevant_documents)

        ranks = {
            name: first_relevant_rank(sources_by_retriever[name], example.relevant_documents)
            for name in RETRIEVERS
        }

        contribution: str | None = None
        rrf_impact: str | None = None
        if example.relevant_documents:
            contribution = classify_contribution(
                ranks["dense"] is not None, ranks["bm25"] is not None
            )
            contribution_counts[contribution] += 1
            rrf_impact = classify_rrf_impact(ranks["dense"], ranks["bm25"], ranks["hybrid"])
            rrf_impact_counts[rrf_impact] += 1

        reference_context_recovered: dict[str, bool | None] = {}
        for name in RETRIEVERS:
            bucket = reference_context_bucket(
                sources_by_retriever[name],
                contents_by_retriever[name],
                example.relevant_documents,
                example.reference_contexts,
            )
            if bucket is not None:
                reference_context_buckets[name].append(bucket)
            reference_context_recovered[name] = (
                {"A": True, "B": False}.get(bucket) if bucket is not None else None
            )

        per_example.append(
            {
                "question": example.question,
                "relevant_documents": example.relevant_documents,
                "content_type": example.content_type,
                "question_type": example.question_type,
                "difficulty": example.difficulty,
                "requires_relationship_expansion": example.requires_relationship_expansion,
                "dense_rank": ranks["dense"],
                "bm25_rank": ranks["bm25"],
                "fused_rank": ranks["hybrid"],
                "contribution_bucket": contribution,
                "rrf_impact": rrf_impact,
                "reference_context_recovered": reference_context_recovered,
            }
        )

    metrics_by_retriever = {
        name: _retriever_metrics(per_retriever_sources[name], all_relevant) for name in RETRIEVERS
    }

    reference_context_summary: dict[str, Any] = {}
    for name in RETRIEVERS:
        counts = Counter(reference_context_buckets[name])
        a_count, b_count = counts.get("A", 0), counts.get("B", 0)
        reference_context_summary[name] = {
            "buckets": dict(counts),
            "supporting_context_hit_rate": (
                (a_count / (a_count + b_count)) if (a_count + b_count) else None
            ),
        }

    breakdowns = {
        dimension: _breakdown_by(examples, per_retriever_sources, all_relevant, key_fn)
        for dimension, key_fn in (
            ("content_type", lambda e: e.content_type or _UNCATEGORIZED),
            ("question_type", lambda e: e.question_type or _UNCATEGORIZED),
            ("difficulty", lambda e: e.difficulty or _UNCATEGORIZED),
            (
                "relationship_aware",
                lambda e: "relationship_aware" if e.requires_relationship_expansion else "other",
            ),
        )
    }

    return {
        "dataset_id": dataset_id,
        "num_examples": len(examples),
        "retrieval_k": RETRIEVAL_K,
        "note": (
            "Ranks and Recall/Hit-Rate/MRR are the only cross-retriever comparison "
            "made here -- raw dense cosine-similarity, BM25, and RRF scores are on "
            "different scales and are never compared directly (see fusion.py)."
        ),
        "metrics_by_retriever": metrics_by_retriever,
        "contribution_buckets": dict(contribution_counts),
        "rrf_impact": dict(rrf_impact_counts),
        "reference_context_by_retriever": reference_context_summary,
        "breakdowns": breakdowns,
        "per_example": per_example,
    }


def select_interesting_examples(
    per_example: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    """Pick up to `limit` per-example rows most worth a human reading.

    Prioritizes the narratively interesting `rrf_impact` categories first
    (a retriever being rescued or degraded is more informative than an
    unremarkable "unchanged"), in a fixed priority order, then fills any
    remaining slots with the next-most-interesting category available.

    Parameters
    ----------
    per_example : list[dict[str, Any]]
        `evaluate_attribution`'s `per_example` list.
    limit : int, optional
        Maximum rows to return, by default 5.

    Returns
    -------
    list[dict[str, Any]]
        Up to `limit` entries from `per_example`, in priority order.
    """
    priority = ["rescued", "degraded", "improved", "still_missed", "unchanged"]
    selected: list[dict[str, Any]] = []
    for category in priority:
        if len(selected) >= limit:
            break
        for entry in per_example:
            if len(selected) >= limit:
                break
            if entry["rrf_impact"] == category and entry not in selected:
                selected.append(entry)
    return selected


def run(gold_path: Path, config_path: str | None, dataset_id: str) -> dict[str, Any]:
    """Load config/gold data and run `evaluate_attribution`, with a report header.

    Parameters
    ----------
    gold_path : Path
        Path to a gold JSONL file.
    config_path : str | None
        Override config path, or None to use `config/default.yaml`.
    dataset_id : str
        Namespace to restrict every retrieval to.

    Returns
    -------
    dict[str, Any]
        `evaluate_attribution`'s report, plus `generated_at` and `config`.
    """
    config: AppConfig = load_config(config_path) if config_path else load_config()
    pipeline = RetrievalPipeline(config)
    examples = load_gold_jsonl(gold_path)
    result = evaluate_attribution(pipeline, examples, dataset_id)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "embedding_model": config.embedding.model_name,
            "rrf_k": config.retrieval.hybrid.rrf_k,
            "retrieval_k": RETRIEVAL_K,
        },
        **result,
    }


def main() -> None:
    """CLI entrypoint: parse args, run `run`, and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold JSONL file")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace to restrict retrieval to. Mandatory -- applied as a filter "
        "on every retrieval this run makes.",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--verbose", action="store_true", help="Include per-question detail in the printed report"
    )
    args = parser.parse_args()

    report = run(Path(args.gold), args.config, args.dataset_id)
    if not args.verbose:
        report = {k: v for k, v in report.items() if k != "per_example"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
