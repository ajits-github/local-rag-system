"""Record a rag.eval.retrieval_attribution report as a standalone experiment artifact.

Distinct from scripts/record_experiment.py: retrieval attribution produces
three parallel metric sets (dense/BM25/hybrid) for one config, not the
single flat metric set record_experiment.py's schema expects, so this
writes its own experiments/results/attribution/<id>.json -- a subdirectory
compare_experiments.py's non-recursive `results_dir.glob("*.json")` never
sees, so the standard comparison table/README stay untouched by this --
and logs its own MLflow run with dense_/bm25_/hybrid_-prefixed metrics,
reusing mlflow_logger.build_run_name for a readable run name. Never
touches an existing experiments/results/*.json record.

Usage:
    python -m rag.eval.retrieval_attribution --gold data/eval/techfusion_gold.jsonl \
        --dataset-id techfusion --verbose > /tmp/attribution_report.json
    python scripts/record_attribution_experiment.py \
        --report /tmp/attribution_report.json --experiment-id experiment_016 \
        --label "dense vs BM25 vs hybrid attribution" --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.eval.mlflow_logger import build_run_name  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results" / "attribution"


def _flatten_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Flatten metrics_by_retriever/reference_context_by_retriever for MLflow logging."""
    flat: dict[str, float] = {}
    for retriever, metrics in report["metrics_by_retriever"].items():
        for name, value in metrics.items():
            flat[f"{retriever}_{name.replace('@', '_at_')}"] = value
    for retriever, summary in report["reference_context_by_retriever"].items():
        rate = summary.get("supporting_context_hit_rate")
        if rate is not None:
            flat[f"{retriever}_supporting_context_hit_rate"] = rate
    return flat


def main() -> None:
    """CLI entrypoint: write the attribution record and log an MLflow run for it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", required=True, help="Path to a rag.eval.retrieval_attribution JSON report"
    )
    parser.add_argument("--experiment-id", required=True, help="e.g. experiment_016")
    parser.add_argument("--label", required=True, help="Short human label")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Config used to produce the report (for embedding model / MLflow settings)",
    )
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    config = load_config(args.config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{args.experiment_id}.json"
    record = {
        "experiment_id": args.experiment_id,
        "label": args.label,
        "record_type": "retrieval_attribution",
        **report,
    }
    results_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {args.experiment_id} -> {results_path}")

    if not config.mlflow.enabled:
        return
    import mlflow

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    run_name = build_run_name(
        {
            "experiment_id": args.experiment_id,
            "generation_model": "retrieval-only",
            "prompt_version": "n/a",
            "retrieval_provider": "attribution",
        }
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "experiment_id": args.experiment_id,
                "label": args.label,
                "record_type": "retrieval_attribution",
                "dataset_id": report.get("dataset_id") or "",
                "embedding_model": config.embedding.model_name,
            }
        )
        mlflow.log_param("rrf_k", report["config"]["rrf_k"])
        mlflow.log_param("retrieval_k", report["retrieval_k"])
        mlflow.log_param("num_examples", report["num_examples"])
        for key, value in _flatten_metrics(report).items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(str(results_path))
        print(f"Logged to MLflow run {run.info.run_id} (tracking_uri={config.mlflow.tracking_uri})")


if __name__ == "__main__":
    main()
