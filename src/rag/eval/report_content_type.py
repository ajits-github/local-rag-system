"""Break down a rag.eval.run_eval report by structured-content type.

Calls `rag.eval.run_eval.run` directly. One retrieval(+generation) pass
feeds both the top-line report and this breakdown, so running this never
re-retrieves or re-generates anything. Buckets `per_example` rows via
`eval.content_type.classify_example` and re-aggregates Recall@5/10, Hit
Rate@5/10, MRR (through the existing, unmodified `eval/metrics.py`
functions) and mean latencies per bucket.

Usage:
    python -m rag.eval.report_content_type --gold data/eval/techfusion_gold.jsonl \
        --dataset-id techfusion --kb-root data/knowledge_base \
        > /tmp/techfusion_eval_62_by_content_type.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.config import AppConfig, load_config
from rag.eval.content_type import build_document_content_types, classify_example
from rag.eval.gold_schema import GoldExample, source_matches_relevant
from rag.eval.metrics import mean_hit_rate_at_k, mean_recall_at_k, mean_reciprocal_rank
from rag.eval.run_eval import RECALL_CUTOFFS, run


def _mean(values: list[float]) -> float:
    """Arithmetic mean of `values`, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _example_from_entry(entry: dict[str, Any]) -> GoldExample:
    """Reconstruct the `GoldExample` fields `classify_example` needs from a `per_example` row."""
    return GoldExample(
        question=entry["question"],
        relevant_documents=entry["relevant_documents"],
        question_type=entry.get("question_type"),
        unanswerable=entry.get("unanswerable", False),
    )


def build_breakdown(report: dict[str, Any], data_root: Path, config: AppConfig) -> dict[str, Any]:
    """Group an eval report's per_example rows by content-type bucket and re-aggregate metrics.

    Parameters
    ----------
    report : dict[str, Any]
        Output of `rag.eval.run_eval.run`, which always populates
        `per_example` regardless of the CLI's `--verbose` flag.
    data_root : Path
        Root that `relevant_documents` entries are relative to (the
        parent of the knowledge-base directory); passed to
        `content_type.build_document_content_types`.
    config : AppConfig
        Used to build the content-type classifier's chunker chain.

    Returns
    -------
    dict[str, Any]
        Maps each bucket name to its own `num_examples`/retrieval/
        hit_rate/mrr/latency_ms summary.
    """
    per_example = report["per_example"]
    all_relevant_paths = (p for entry in per_example for p in entry["relevant_documents"])
    document_content_types = build_document_content_types(data_root, all_relevant_paths, config)

    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in per_example:
        bucket = classify_example(_example_from_entry(entry), document_content_types)
        buckets.setdefault(bucket, []).append(entry)

    breakdown: dict[str, Any] = {}
    for bucket, entries in sorted(buckets.items()):
        all_retrieved = [e["retrieved_sources"] for e in entries]
        all_relevant = [e["relevant_documents"] for e in entries]
        summary: dict[str, Any] = {
            "num_examples": len(entries),
            "retrieval": {
                f"recall@{k}": mean_recall_at_k(
                    all_retrieved, all_relevant, k, source_matches_relevant
                )
                for k in RECALL_CUTOFFS
            },
            "hit_rate": {
                f"hit_rate@{k}": mean_hit_rate_at_k(
                    all_retrieved, all_relevant, k, source_matches_relevant
                )
                for k in RECALL_CUTOFFS
            },
            "mrr": mean_reciprocal_rank(all_retrieved, all_relevant, source_matches_relevant),
        }
        retrieval_ms = [e["retrieval_ms"] for e in entries if "retrieval_ms" in e]
        generation_ms = [e["generation_ms"] for e in entries if "generation_ms" in e]
        total_ms = [e["total_ms"] for e in entries if "total_ms" in e]
        if retrieval_ms or generation_ms or total_ms:
            summary["latency_ms"] = {
                "retrieval_mean": _mean(retrieval_ms),
                "generation_mean": _mean(generation_ms),
                "total_mean": _mean(total_ms),
            }
        breakdown[bucket] = summary
    return breakdown


def main() -> None:
    """CLI entrypoint: run the base eval, break it down by content type, print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold JSONL file")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace to restrict retrieval to (e.g. 'techfusion')",
    )
    parser.add_argument(
        "--kb-root",
        required=True,
        help="Knowledge-base directory that relevant_documents entries are relative to",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Retrieval metrics only -- skips LLM generation, so no per-bucket latency.",
    )
    args = parser.parse_args()

    report = run(
        Path(args.gold), args.config, args.dataset_id, run_generation=not args.skip_generation
    )
    config = load_config(args.config) if args.config else load_config()
    data_root = Path(args.kb_root).parent
    breakdown = build_breakdown(report, data_root, config)

    print(json.dumps({"dataset_id": args.dataset_id, "by_content_type": breakdown}, indent=2))


if __name__ == "__main__":
    main()
