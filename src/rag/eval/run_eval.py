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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from rag.config import AppConfig, load_config
from rag.eval.answer_quality import KeywordOverlapScorer
from rag.eval.gold_schema import (
    GoldExample,
    load_gold_jsonl,
    reference_context_is_supported,
    source_matches_relevant,
)
from rag.eval.metrics import mean_hit_rate_at_k, mean_recall_at_k, mean_reciprocal_rank
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import SearchResult

# Broadest cutoff fetched per query; @5 is sliced from the same fetch so
# each gold question only needs one retrieval call for both cutoffs.
RETRIEVAL_K = 10
RECALL_CUTOFFS = (5, 10)

# Gold rows predating the multimodal milestone have no authored content_type.
_UNCATEGORIZED = "uncategorized"

# Reused from scripts/validate_gold_file.py's keyword-overlap threshold for
# consistency; see KeywordOverlapScorer's own documented caveat -- a crude
# heuristic, not a semantic-correctness judge.
_VISION_BEHAVIOR_QUALITY_THRESHOLD = 0.15

# Deliberately small and literal (not exhaustive NLP) -- a triage aid for
# hand-inspection (see PROJECT_JOURNAL.md), not a claim of semantic
# understanding of refusal. Calibrated against techfusion_gold.jsonl's own
# unanswerable expected_answer phrasing as well as generic refusal wording.
_REFUSAL_PHRASES = (
    "don't know",
    "do not know",
    "cannot determine",
    "can't determine",
    "not shown",
    "not provided",
    "no information",
    "not available",
    "does not contain",
    "doesn't contain",
    "does not provide",
    "provide no",
    "provides no",
    "does not name",
    "does not describe",
    "not described",
    "not included",
    "cannot be answered",
    "not be answered",
    "unavailable",
    "unable to determine",
    "not specified",
    "not mentioned",
    "not applicable",
    "cannot answer",
    "can't answer",
    "no answer should be inferred",
)


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
        "retrieval_provider": config.retrieval.provider,
        "retrieval_top_k": config.retrieval.top_k,
        "rerank_top_n": config.retrieval.rerank_top_n,
        "relationship_expansion_enabled": config.retrieval.relationship_expansion.enabled,
        "vision_provider": config.vision.provider,
    }


def _resolve_asset_path(source: str, source_anchor: str) -> str:
    """Resolve a chunk's `source_anchor` (e.g. "images/x.png") relative to its `source` document."""
    return str(PurePosixPath(source.replace("\\", "/")).parent / source_anchor)


def _image_hit(results: list[SearchResult], relevant_images: list[str]) -> bool:
    """Whether any `results` chunk's resolved asset path matches a `relevant_images` entry.

    Reuses `source_matches_relevant`'s path-suffix matching (the same rule
    already used for `relevant_documents`) against each result's
    `source_anchor` resolved relative to its own document path -- valid in
    text-only mode too, since it only asks "was the right image asset
    surfaced at all," independent of whether a vision description exists.
    """
    if not relevant_images:
        return False
    for r in results:
        anchor = r.chunk.metadata.source_anchor
        if not anchor:
            continue
        resolved = _resolve_asset_path(r.chunk.metadata.source, anchor)
        if any(source_matches_relevant(resolved, img) for img in relevant_images):
            return True
    return False


def _looks_like_refusal(answer_text: str) -> bool:
    """Whether `answer_text` contains one of `_REFUSAL_PHRASES` (case-insensitive)."""
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def _classify_vision_behavior(
    example: GoldExample, answer_text: str, answer_quality: float | None
) -> str:
    """Categorize one `requires_vision=True` example's text-only-mode answer.

    One of `"correct_refusal"` / `"hallucinated_answer"` /
    `"caption_leak_success"` / `"incorrect_or_missing"` -- see
    `evaluate`'s docstring for what each means. This is a heuristic triage
    aid for hand-inspection (see PROJECT_JOURNAL.md's failure-analysis
    entries), not an automated semantic-correctness judgment:
    `_looks_like_refusal` is a literal phrase match, and `answer_quality`
    is `KeywordOverlapScorer`'s crude keyword-overlap heuristic.
    `caption_leak_success` covers both a legitimate answer from caption
    text (when `example.content_type == "caption_answerable"`) and a
    genuine accidental leak (any other content_type) -- both look
    identical by this deterministic check, so distinguishing them is left
    to hand-inspection using the reported `content_type`.

    Parameters
    ----------
    example : GoldExample
        The `requires_vision=True` gold example.
    answer_text : str
        The generated answer.
    answer_quality : float | None
        `KeywordOverlapScorer` score against `expected_answer`, if computed.

    Returns
    -------
    str
        One of the four categories above.
    """
    refusal = _looks_like_refusal(answer_text)
    if example.unanswerable:
        return "correct_refusal" if refusal else "hallucinated_answer"
    if refusal:
        return "incorrect_or_missing"
    if answer_quality is not None and answer_quality >= _VISION_BEHAVIOR_QUALITY_THRESHOLD:
        return "caption_leak_success"
    return "incorrect_or_missing"


def _content_type_breakdown(
    examples: list[GoldExample],
    all_retrieved_sources: list[list[str]],
    all_relevant: list[list[str]],
) -> dict[str, dict[str, Any]]:
    """Bucket examples by authored `content_type`; report Recall@5/@10/hit-rate@5 per bucket.

    Distinct from `eval/content_type.py`'s chunker-derived document buckets
    (table/code_configuration/chart/prose/multi_hop/unanswerable) -- this
    uses the gold file's own authored ground-truth question category
    (text_only/table/chart/caption_answerable/image_only/text_plus_image/
    relationship_aware/unanswerable_visual/architecture_diagram/
    table_image/...), grouping rows with no such field under
    `_UNCATEGORIZED` (pre-multimodal-milestone gold rows).

    Parameters
    ----------
    examples : list[GoldExample]
        All evaluated gold examples.
    all_retrieved_sources : list[list[str]]
        Per-example broad-retrieval source lists, same order as `examples`.
    all_relevant : list[list[str]]
        Per-example `relevant_documents`, same order as `examples`.

    Returns
    -------
    dict[str, dict[str, Any]]
        One entry per bucket: `{"count", "recall@5", "recall@10", "hit_rate@5"}`.
    """
    bucket_indices: dict[str, list[int]] = {}
    for i, example in enumerate(examples):
        bucket_indices.setdefault(example.content_type or _UNCATEGORIZED, []).append(i)

    breakdown: dict[str, dict[str, Any]] = {}
    for bucket, indices in sorted(bucket_indices.items()):
        retrieved = [all_retrieved_sources[i] for i in indices]
        relevant = [all_relevant[i] for i in indices]
        breakdown[bucket] = {
            "count": len(indices),
            "recall@5": mean_recall_at_k(retrieved, relevant, 5, source_matches_relevant),
            "recall@10": mean_recall_at_k(retrieved, relevant, 10, source_matches_relevant),
            "hit_rate@5": mean_hit_rate_at_k(retrieved, relevant, 5, source_matches_relevant),
        }
    return breakdown


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
        Report with `retrieval`, `hit_rate`, `mrr`, `content_type_breakdown`,
        `reference_context_analysis` (the A/B/C supporting-context-hit
        buckets), and (when applicable) `relevant_image_hit_rate`/
        `relationship_expansion_contribution_rate`/(if `run_generation`)
        `vision_behavior_breakdown`/`latency_ms`/`answer_quality` keys,
        plus `per_example` detail. See `_content_type_breakdown`,
        `_classify_vision_behavior`, and `reference_context_is_supported`
        for what each new metric means and its documented limitations --
        all are deterministic heuristics, not semantic-correctness judges.
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
    reference_context_buckets: list[str] = []
    image_hit_records: list[bool] = []
    expansion_contribution_records: list[bool] = []
    vision_behavior_records: list[str] = []
    per_example: list[dict[str, Any]] = []

    for example in examples:
        # Broad retrieval (top 10, un-truncated by the reranker) purely to
        # score retrieval quality at multiple cutoffs -- decoupled from the
        # production-config latency measured via pipeline.answer() below.
        # dataset_filter is mandatory here: this is the isolation guarantee.
        # Also the source of the reference-context/image-hit metrics below
        # (K in the design review) -- no extra retrieval call needed for
        # those, since `origin`/content on these same results already
        # distinguish directly-retrieved from relationship-expanded chunks.
        retrieval_results = pipeline.retrieve(
            example.question,
            filters=dataset_filter,
            top_k=RETRIEVAL_K,
            rerank_top_n=RETRIEVAL_K,
        )
        retrieved_sources = [r.chunk.metadata.source for r in retrieval_results]
        all_retrieved_sources.append(retrieved_sources)
        all_relevant.append(example.relevant_documents)

        reference_context_bucket: str | None = None
        if example.relevant_documents:
            doc_retrieved = any(
                source_matches_relevant(s, rd)
                for s in retrieved_sources
                for rd in example.relevant_documents
            )
            if not doc_retrieved:
                reference_context_bucket = "C"
            elif not example.reference_contexts:
                reference_context_bucket = "not_applicable"
            else:
                all_contents = [r.chunk.content for r in retrieval_results]
                context_hit = all(
                    reference_context_is_supported(ref, all_contents)
                    for ref in example.reference_contexts
                )
                reference_context_bucket = "A" if context_hit else "B"
            reference_context_buckets.append(reference_context_bucket)

        if example.requires_relationship_expansion and example.reference_contexts:
            retrieved_only_contents = [
                r.chunk.content for r in retrieval_results if r.origin == "retrieved"
            ]
            all_contents = [r.chunk.content for r in retrieval_results]
            pre_expansion_hit = all(
                reference_context_is_supported(ref, retrieved_only_contents)
                for ref in example.reference_contexts
            )
            overall_hit = all(
                reference_context_is_supported(ref, all_contents)
                for ref in example.reference_contexts
            )
            expansion_contribution_records.append(overall_hit and not pre_expansion_hit)

        relevant_image_hit: bool | None = None
        if example.relevant_images:
            relevant_image_hit = _image_hit(retrieval_results, example.relevant_images)
            image_hit_records.append(relevant_image_hit)

        entry: dict[str, Any] = {
            "question": example.question,
            "question_type": example.question_type,
            "difficulty": example.difficulty,
            "unanswerable": example.unanswerable,
            "relevant_documents": example.relevant_documents,
            "retrieved_sources": retrieved_sources,
            "content_type": example.content_type,
            "requires_vision": example.requires_vision,
            "requires_relationship_expansion": example.requires_relationship_expansion,
            "reference_context_bucket": reference_context_bucket,
            "relevant_image_hit": relevant_image_hit,
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
                # Production-config sources-with-content (this answer()
                # call's own retrieve, at real rerank_top_n) -- distinct
                # from `retrieved_sources` above, which comes from the
                # broader top-10 fetch used for Recall@10. Consumed by
                # rag.eval.run_ragas_eval for RAGAS's retrieved_contexts.
                generation_sources=result["sources"],
                retrieval_ms=result["retrieval_ms"],
                generation_ms=result["generation_ms"],
                total_ms=result["total_ms"],
            )
            quality: float | None = None
            if scorer is not None and example.expected_answer:
                quality = scorer.score(example.question, result["answer"], example.expected_answer)
                entry["answer_quality"] = quality
                bucket = (
                    unanswerable_quality_scores
                    if example.unanswerable
                    else answerable_quality_scores
                )
                bucket.append(quality)
            if example.requires_vision:
                behavior = _classify_vision_behavior(example, result["answer"], quality)
                entry["vision_behavior"] = behavior
                vision_behavior_records.append(behavior)

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
        "content_type_breakdown": _content_type_breakdown(
            examples, all_retrieved_sources, all_relevant
        ),
    }

    bucket_counts = Counter(reference_context_buckets)
    a_count, b_count = bucket_counts.get("A", 0), bucket_counts.get("B", 0)
    report["reference_context_analysis"] = {
        "note": (
            "A = relevant document retrieved AND supporting reference_contexts found "
            "(verbatim substring match, see reference_context_is_supported); "
            "B = relevant document retrieved but supporting context missed; "
            "C = relevant document missed entirely; not_applicable = the gold example "
            "has relevant_documents but no authored reference_contexts to check against. "
            "supporting_context_hit_rate = A / (A + B), i.e. among questions where the "
            "right document WAS found, how often the specific supporting passage was too."
        ),
        "buckets": dict(bucket_counts),
        "supporting_context_hit_rate": (
            (a_count / (a_count + b_count)) if (a_count + b_count) else None
        ),
    }

    if image_hit_records:
        report["relevant_image_hit_rate"] = {
            "note": (
                "Among gold examples with a non-empty relevant_images, whether any "
                "retrieved chunk's resolved image asset path matched one of them -- "
                "meaningful in text-only mode too (checks the caption-based image "
                "element was surfaced, independent of whether it was sufficient to answer)."
            ),
            "count": len(image_hit_records),
            "hit_rate": _mean([1.0 if hit else 0.0 for hit in image_hit_records]),
        }

    if expansion_contribution_records:
        report["relationship_expansion_contribution_rate"] = {
            "note": (
                "Among requires_relationship_expansion=True examples with authored "
                "reference_contexts, the fraction where an origin='expanded' chunk "
                "supplied supporting context that the pre-expansion (origin='retrieved') "
                "set alone did not -- demonstrates expansion contributed evidence, not "
                "just that it fired. 0.0 when relationship_expansion.enabled=false, by "
                "construction (no chunk ever has origin='expanded')."
            ),
            "count": len(expansion_contribution_records),
            "rate": _mean([1.0 if c else 0.0 for c in expansion_contribution_records]),
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
        if vision_behavior_records:
            report["vision_behavior_breakdown"] = {
                "note": (
                    "Heuristic triage of requires_vision=True examples' text-only-mode "
                    "answers -- see _classify_vision_behavior's docstring for exactly what "
                    "each category means and its limitations (phrase-match refusal "
                    "detection + KeywordOverlapScorer, not semantic judgment). "
                    "correct_refusal/hallucinated_answer apply to gold-unanswerable "
                    "questions; caption_leak_success/incorrect_or_missing apply to the "
                    "rest -- cross-reference each per_example row's content_type to tell "
                    "a legitimate caption_answerable success from an accidental leak."
                ),
                "counts": dict(Counter(vision_behavior_records)),
                "total": len(vision_behavior_records),
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
