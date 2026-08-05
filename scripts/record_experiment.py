"""Register a rag.eval.run_eval report as a comparable experiment record.

Reads the config that actually produced a given eval run (authoritative
for embedding/chunking/reranker/generation/vectorstore settings) plus the
eval report's own metrics, and writes:
  - experiments/results/<experiment-id>.json  (flat, comparable record)
  - experiments/configs/<experiment-id>.yaml  (config snapshot)

This never re-runs evaluation -- it only records an existing report. Run
`rag.eval.run_eval --verbose` first, then point this at its output.

Usage:
    python -m rag.eval.run_eval --gold data/eval/techfusion_gold.jsonl \
        --dataset-id techfusion --verbose > /tmp/eval_report.json
    python scripts/record_experiment.py --eval-output /tmp/eval_report.json \
        --experiment-id experiment_002 --label "cross-encoder reranker" \
        --config config/default.yaml
    python scripts/compare_experiments.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, AppConfig, load_config  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _reranker_model(config: AppConfig) -> str | None:
    if config.reranker.provider == "cross_encoder":
        return config.reranker.cross_encoder.model_name
    if config.reranker.provider == "cohere":
        return config.reranker.cohere.model_name
    return None


def build_experiment_record(
    eval_report: dict, config: AppConfig, experiment_id: str, label: str
) -> dict:
    """Flatten an eval report + the config that produced it into the fixed
    schema experiments/ compares on. Config fields are read from `config`
    (the actual file used), not from the report's own embedded summary,
    so this works even against older reports that didn't capture every
    field (e.g. vector_store, reranker_model)."""
    retrieval = eval_report.get("retrieval", {})
    hit_rate = eval_report.get("hit_rate", {})
    latency = eval_report.get("latency_ms", {})
    answer_quality = eval_report.get("answer_quality", {})

    return {
        "experiment_id": experiment_id,
        "label": label,
        "timestamp": eval_report.get("generated_at"),
        "generation_model": config.generation.model_name,
        "embedding_model": config.embedding.model_name,
        "vector_store": config.vectorstore.provider,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "reranker_provider": config.reranker.provider,
        "reranker_model": _reranker_model(config),
        "retrieval_top_k": config.retrieval.top_k,
        "rerank_top_n": config.retrieval.rerank_top_n,
        "dataset_id": eval_report.get("dataset_id"),
        "recall_at_5": retrieval.get("recall@5"),
        "recall_at_10": retrieval.get("recall@10"),
        "hit_rate_at_5": hit_rate.get("hit_rate@5"),
        "hit_rate_at_10": hit_rate.get("hit_rate@10"),
        "mrr": eval_report.get("mrr"),
        "answer_quality": answer_quality.get("mean_overall"),
        "retrieval_latency_ms": latency.get("retrieval_mean"),
        "generation_latency_ms": latency.get("generation_mean"),
        "total_latency_ms": latency.get("total_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-output", required=True, help="Path to a rag.eval.run_eval JSON report"
    )
    parser.add_argument("--experiment-id", required=True, help="e.g. experiment_002")
    parser.add_argument(
        "--label", required=True, help="Short human label, e.g. 'cross-encoder reranker'"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Config file that produced this eval run (snapshotted alongside the record)",
    )
    args = parser.parse_args()

    eval_report = json.loads(Path(args.eval_output).read_text(encoding="utf-8"))
    config = load_config(args.config)
    record = build_experiment_record(eval_report, config, args.experiment_id, args.label)

    results_dir = EXPERIMENTS_DIR / "results"
    configs_dir = EXPERIMENTS_DIR / "configs"
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{args.experiment_id}.json"
    results_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    shutil.copy(args.config, configs_dir / f"{args.experiment_id}.yaml")

    print(f"Recorded {args.experiment_id} -> {results_path}")


if __name__ == "__main__":
    main()
