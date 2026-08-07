"""Register a rag.eval.run_eval report as a comparable experiment record.

Reads the config that actually produced a given eval run (authoritative
for embedding/chunking/reranker/generation/vectorstore settings) plus the
eval report's own metrics, and writes:
  - experiments/results/<experiment-id>.json  (flat, comparable record)
  - experiments/configs/<experiment-id>.yaml  (config snapshot)
  - an MLflow run (params/metrics/artifacts), unless config.mlflow.enabled
    is false -- see rag.eval.mlflow_logger.

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
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, AppConfig, load_config  # noqa: E402
from rag.eval.mlflow_logger import log_experiment  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _reranker_model(config: AppConfig) -> str | None:
    """Return the active reranker's model name, or None for provider 'none'."""
    if config.reranker.provider == "cross_encoder":
        return config.reranker.cross_encoder.model_name
    if config.reranker.provider == "cohere":
        return config.reranker.cohere.model_name
    return None


def build_experiment_record(
    eval_report: dict[str, Any], config: AppConfig, experiment_id: str, label: str
) -> dict[str, Any]:
    """Flatten an eval report and its config into the fixed experiments/ record schema.

    Config fields are read from `config` (the actual file used), not from
    the report's own embedded summary, so this works even against older
    reports that didn't capture every field (e.g. vector_store, reranker_model).

    Parameters
    ----------
    eval_report : dict[str, Any]
        Parsed JSON output of `rag.eval.run_eval --verbose`.
    config : AppConfig
        The config that produced `eval_report`.
    experiment_id : str
        Identifier for this experiment, e.g. ``"experiment_002"``.
    label : str
        Short human-readable label, e.g. ``"cross-encoder reranker"``.

    Returns
    -------
    dict[str, Any]
        Flat record matching the `experiments/results/*.json` schema.
    """
    retrieval = eval_report.get("retrieval", {})
    hit_rate = eval_report.get("hit_rate", {})
    latency = eval_report.get("latency_ms", {})
    answer_quality = eval_report.get("answer_quality", {})
    ragas = eval_report.get("ragas", {})
    ragas_aggregate = ragas.get("aggregate", {})
    ragas_judge = ragas.get("judge", {})

    return {
        "experiment_id": experiment_id,
        "label": label,
        "timestamp": eval_report.get("generated_at"),
        "generation_model": config.generation.model_name,
        "embedding_model": config.embedding.model_name,
        "vector_store": config.vectorstore.provider,
        "chunking_provider": config.chunking.provider,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "reranker_provider": config.reranker.provider,
        "reranker_model": _reranker_model(config),
        "prompt_id": config.generation.prompt.id,
        "prompt_version": config.generation.prompt.version,
        "prompt_file_checksum": hashlib.sha256(
            config.prompt_template_path().read_bytes()
        ).hexdigest(),
        "retrieval_provider": config.retrieval.provider,
        "retrieval_top_k": config.retrieval.top_k,
        "rerank_top_n": config.retrieval.rerank_top_n,
        "rrf_k": (config.retrieval.hybrid.rrf_k if config.retrieval.provider == "hybrid" else None),
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
        "ragas_faithfulness": ragas_aggregate.get("faithfulness"),
        "ragas_answer_relevancy": ragas_aggregate.get("answer_relevancy"),
        "ragas_context_precision": ragas_aggregate.get("context_precision"),
        "ragas_context_recall": ragas_aggregate.get("context_recall"),
        "ragas_answer_correctness": ragas_aggregate.get("answer_correctness"),
        "ragas_judge_provider": ragas_judge.get("provider"),
        "ragas_judge_model": ragas_judge.get("model_name"),
    }


def main() -> None:
    """CLI entrypoint: build a record from an eval report and write it to disk."""
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
    config_snapshot_path = configs_dir / f"{args.experiment_id}.yaml"
    shutil.copy(args.config, config_snapshot_path)

    print(f"Recorded {args.experiment_id} -> {results_path}")

    run_id = log_experiment(
        record,
        config.mlflow,
        artifact_paths=[Path(args.eval_output), results_path, config_snapshot_path],
    )
    if run_id:
        print(f"Logged to MLflow run {run_id} (tracking_uri={config.mlflow.tracking_uri})")


if __name__ == "__main__":
    main()
