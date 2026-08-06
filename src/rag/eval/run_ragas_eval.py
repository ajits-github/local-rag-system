"""Run RAGAS generation-quality evaluation on top of a base rag.eval.run_eval report.

Opt-in and additive: reuses `run_eval.evaluate()` (sliced to
`--sample-size` examples) for the standard report, then adds a `"ragas"`
key with faithfulness/answer_relevancy/context_precision/context_recall/
answer_correctness scores from an independently configured judge LLM
(`config.judge` — never defaults to the generation models). Requires the
`ragas` extra (`pip install .[ragas]`, plus `.[anthropic]` for that hosted
provider).

Usage:
    python -m rag.eval.run_ragas_eval --gold data/eval/sample_gold.jsonl \
        --dataset-id sample_docs --verbose > /tmp/ragas_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import AppConfig, load_config
from rag.eval import ragas_scorer
from rag.eval.gold_schema import GoldExample, load_gold_jsonl
from rag.eval.run_eval import _config_summary, evaluate
from rag.factory import build_embedder, build_judge_llm
from rag.generation.base import LLM
from rag.retrieval.pipeline import RetrievalPipeline

DEFAULT_SAMPLE_SIZE = 15


def _build_rows(
    examples: list[GoldExample], per_example: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Zip sliced `GoldExample`s with `evaluate()`'s per_example entries into scoring rows.

    Skips examples with no `expected_answer` — RAGAS's `reference`-requiring
    metrics can't function without one.

    Parameters
    ----------
    examples : list[GoldExample]
        The (already-sliced) gold examples run through `evaluate()`.
    per_example : list[dict[str, Any]]
        `evaluate()`'s `per_example` output, same order/length as `examples`.

    Returns
    -------
    tuple[list[dict[str, Any]], int]
        ``(rows, num_skipped)``.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for i, (example, entry) in enumerate(zip(examples, per_example, strict=True)):
        if not example.expected_answer:
            skipped += 1
            continue
        rows.append(
            {
                "question_index": i,
                "question": example.question,
                "unanswerable": example.unanswerable,
                "answer": entry["answer"],
                "contexts": [s["content"] for s in entry["generation_sources"]],
                "expected_answer": example.expected_answer,
            }
        )
    return rows, skipped


def _judge_summary(config: AppConfig, judge_llm: LLM) -> dict[str, Any]:
    """Build the report's `judge` object: provider/model/estimated_api_usage.

    `estimated_api_usage` is ``None`` for the `ollama` provider (no API
    cost); for hosted providers it's read off the judge LLM's own
    call-count/token-usage tracking attributes.
    """
    provider = config.judge.provider
    model_name = {
        "openai": config.judge.openai.model_name,
        "anthropic": config.judge.anthropic.model_name,
        "ollama": config.judge.ollama.model_name,
    }[provider]
    usage = None
    if provider in {"openai", "anthropic"}:
        usage = {
            "call_count": judge_llm.call_count,  # type: ignore[attr-defined]
            "input_tokens": judge_llm.input_tokens,  # type: ignore[attr-defined]
            "output_tokens": judge_llm.output_tokens,  # type: ignore[attr-defined]
        }
    return {"provider": provider, "model_name": model_name, "estimated_api_usage": usage}


def run_ragas(
    gold_path: Path,
    config_path: str | None,
    dataset_id: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Load config/gold data, run the base eval on a sample, then add RAGAS scores.

    Parameters
    ----------
    gold_path : Path
        Path to a gold JSONL file.
    config_path : str | None
        Override config path, or None to use `config/default.yaml`.
    dataset_id : str
        Namespace to restrict every retrieval to.
    sample_size : int, optional
        Number of gold examples (from the start of the file) to score, by
        default 15 — validate cost/latency/quality before scaling up.

    Returns
    -------
    dict[str, Any]
        The same shape as `run_eval.run()`'s report, plus a `"ragas"` key.
    """
    config = load_config(config_path) if config_path else load_config()
    pipeline = RetrievalPipeline(config)
    examples = load_gold_jsonl(gold_path)[:sample_size]
    base_result = evaluate(pipeline, examples, dataset_id, run_generation=True)

    judge_llm = build_judge_llm(config)
    embedder = build_embedder(config)
    rows, num_skipped = _build_rows(examples, base_result["per_example"])
    ragas_result = ragas_scorer.score(rows, judge_llm, embedder)
    ragas_result.update(
        {
            "prompt_id": config.generation.prompt.id,
            "prompt_version": config.generation.prompt.version,
            "judge": _judge_summary(config, judge_llm),
            "sample_size": len(examples),
            "num_scored": len(rows),
            "num_skipped_missing_expected_answer": num_skipped,
        }
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": _config_summary(config),
        **base_result,
        "ragas": ragas_result,
    }


def main() -> None:
    """CLI entrypoint: parse args, run `run_ragas`, and print the JSON report."""
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
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of gold examples to score, by default {DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Include per-question detail in the printed report"
    )
    args = parser.parse_args()

    report = run_ragas(Path(args.gold), args.config, args.dataset_id, args.sample_size)
    if not args.verbose:
        report = {k: v for k, v in report.items() if k != "per_example"}
        report["ragas"] = {k: v for k, v in report["ragas"].items() if k != "per_question"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
