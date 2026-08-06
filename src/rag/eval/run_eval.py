"""Run retrieval + generation evaluation against a gold JSONL dataset.

Works with any gold file matching GoldExample's schema (question /
expected_answer / relevant_documents / question_type / difficulty /
unanswerable) and any knowledge base, regardless of what root path it was
ingested from -- nothing here is specific to a particular dataset.

--dataset-id is mandatory and is injected as a `dataset_id` filter on every
retrieval this runner makes -- an evaluation can never silently retrieve
chunks from a different dataset than the one it's meant to be scoring
(e.g. a TechFusion eval accidentally pulling in data/sample_docs chunks).

Usage:
    python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl --dataset-id techfusion
    python -m rag.eval.run_eval --gold data/eval/sample_gold.jsonl --dataset-id sample_docs
    python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl \
        --dataset-id techfusion --skip-generation
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import AppConfig, load_config
from rag.eval.answer_quality import KeywordOverlapScorer
from rag.eval.gold_schema import GoldExample, load_gold_jsonl, source_matches_relevant
from rag.eval.metrics import mean_hit_rate_at_k, mean_recall_at_k, mean_reciprocal_rank
from rag.retrieval.pipeline import RetrievalPipeline

# Broadest cutoff fetched per query; @5 is sliced from the same fetch so
# each gold question only needs one retrieval call for both cutoffs.
RETRIEVAL_K = 10
RECALL_CUTOFFS = (5, 10)


def _mean(values: list[float]) -> float:
    """Arithmetic mean of `values`, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _config_summary(config: AppConfig) -> dict[str, Any]:
    """Snapshot the provider choices that produced this report.

    Makes a saved baseline file self-describing, so a later comparison
    (e.g. scripts/render_benchmarks.py) doesn't depend on remembering
    which config was active when it was generated.
    """
    return {
        "embedding_model": config.embedding.model_name,
        "chunking_provider": config.chunking.provider,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "reranker_provider": config.reranker.provider,
        "generation_model": config.generation.model_name,
        "prompt_id": config.generation.prompt.id,
        "prompt_version": config.generation.prompt.version,
        "retrieval_top_k": config.retrieval.top_k,
        "rerank_top_n": config.retrieval.rerank_top_n,
    }


def evaluate(
    pipeline: RetrievalPipeline,
    examples: list[GoldExample],
    dataset_id: str,
    run_generation: bool = True,
) -> dict[str, Any]:
    """Run retrieval (and optionally generation) over `examples` and score the results.

    Every retrieval call is restricted to `dataset_id` via a mandatory
    filter, so an evaluation can never silently score chunks retrieved
    from a different dataset.

    Parameters
    ----------
    pipeline : RetrievalPipeline
        Pipeline to evaluate.
    examples : list[GoldExample]
        Gold questions to run.
    dataset_id : str
        Namespace to restrict every retrieval to.
    run_generation : bool, optional
        If True (default), also runs `pipeline.answer` for latency and
        answer-quality metrics; if False, only retrieval metrics are
        computed.

    Returns
    -------
    dict[str, Any]
        Report with `retrieval`, `hit_rate`, `mrr`, and (if
        `run_generation`) `latency_ms`/`answer_quality` keys, plus
        `per_example` detail.
    """
    scorer = KeywordOverlapScorer() if run_generation else None
    dataset_filter = {"dataset_id": dataset_id}

    all_retrieved_sources: list[list[str]] = []
    all_relevant: list[list[str]] = []
    retrieval_ms_values: list[float] = []
    generation_ms_values: list[float] = []
    total_ms_values: list[float] = []
    answerable_quality_scores: list[float] = []
    unanswerable_quality_scores: list[float] = []
    per_example: list[dict[str, Any]] = []

    for example in examples:
        # Broad retrieval (top 10, un-truncated by the reranker) purely to
        # score retrieval quality at multiple cutoffs -- decoupled from the
        # production-config latency measured via pipeline.answer() below.
        # dataset_filter is mandatory here: this is the isolation guarantee.
        retrieval_results = pipeline.retrieve(
            example.question,
            filters=dataset_filter,
            top_k=RETRIEVAL_K,
            rerank_top_n=RETRIEVAL_K,
        )
        retrieved_sources = [r.chunk.metadata.source for r in retrieval_results]
        all_retrieved_sources.append(retrieved_sources)
        all_relevant.append(example.relevant_documents)

        entry: dict[str, Any] = {
            "question": example.question,
            "question_type": example.question_type,
            "difficulty": example.difficulty,
            "unanswerable": example.unanswerable,
            "relevant_documents": example.relevant_documents,
            "retrieved_sources": retrieved_sources,
        }

        if run_generation:
            # Production-config answer (real rerank_top_n, real prompt and
            # generation call) for latency + answer-quality measurement --
            # same mandatory dataset_id filter applied.
            result = pipeline.answer(example.question, filters=dataset_filter)
            retrieval_ms_values.append(result["retrieval_ms"])
            generation_ms_values.append(result["generation_ms"])
            total_ms_values.append(result["total_ms"])
            entry.update(
                answer=result["answer"],
                retrieval_ms=result["retrieval_ms"],
                generation_ms=result["generation_ms"],
                total_ms=result["total_ms"],
            )
            if scorer is not None and example.expected_answer:
                quality = scorer.score(example.question, result["answer"], example.expected_answer)
                entry["answer_quality"] = quality
                bucket = (
                    unanswerable_quality_scores
                    if example.unanswerable
                    else answerable_quality_scores
                )
                bucket.append(quality)

        per_example.append(entry)

    report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "num_examples": len(examples),
        "retrieval": {
            f"recall@{k}": mean_recall_at_k(
                all_retrieved_sources, all_relevant, k, source_matches_relevant
            )
            for k in RECALL_CUTOFFS
        },
        "hit_rate": {
            f"hit_rate@{k}": mean_hit_rate_at_k(
                all_retrieved_sources, all_relevant, k, source_matches_relevant
            )
            for k in RECALL_CUTOFFS
        },
        "mrr": mean_reciprocal_rank(all_retrieved_sources, all_relevant, source_matches_relevant),
    }

    if run_generation:
        report["latency_ms"] = {
            "retrieval_mean": _mean(retrieval_ms_values),
            "generation_mean": _mean(generation_ms_values),
            "total_mean": _mean(total_ms_values),
        }
        all_quality = answerable_quality_scores + unanswerable_quality_scores
        report["answer_quality"] = {
            "note": (
                "Keyword-overlap heuristic (eval/answer_quality.py) -- a crude "
                "placeholder, not a faithfulness/correctness judge. Particularly "
                "unreliable on unanswerable questions, where a correct refusal "
                "may share few keywords with the reference 'no answer' text. "
                "See README Roadmap (RAGAS) for a real answer-quality suite."
            ),
            "mean_overall": _mean(all_quality),
            "mean_answerable": _mean(answerable_quality_scores),
            "mean_unanswerable": _mean(unanswerable_quality_scores),
        }

    report["per_example"] = per_example
    return report


def run(
    gold_path: Path, config_path: str | None, dataset_id: str, run_generation: bool = True
) -> dict[str, Any]:
    """Load config and gold data, run `evaluate`, and attach a report header.

    Parameters
    ----------
    gold_path : Path
        Path to a gold JSONL file.
    config_path : str | None
        Override config path, or None to use `config/default.yaml`.
    dataset_id : str
        Namespace to restrict every retrieval to.
    run_generation : bool, optional
        Passed through to `evaluate`, by default True.

    Returns
    -------
    dict[str, Any]
        `evaluate`'s report, plus `generated_at` and `config` keys.
    """
    config = load_config(config_path) if config_path else load_config()
    pipeline = RetrievalPipeline(config)
    examples = load_gold_jsonl(gold_path)
    result = evaluate(pipeline, examples, dataset_id, run_generation=run_generation)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": _config_summary(config),
        **result,
    }


def main() -> None:
    """CLI entrypoint: parse args, run `evaluate`, and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Path to a gold JSONL file")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Namespace to restrict retrieval to (e.g. 'techfusion'). Mandatory -- "
        "applied as a filter on every retrieval this run makes.",
    )
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Retrieval metrics only -- skips LLM generation, latency, and answer-quality scoring.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Include per-question detail in the printed report"
    )
    args = parser.parse_args()

    report = run(
        Path(args.gold), args.config, args.dataset_id, run_generation=not args.skip_generation
    )
    if not args.verbose:
        report = {k: v for k, v in report.items() if k != "per_example"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
