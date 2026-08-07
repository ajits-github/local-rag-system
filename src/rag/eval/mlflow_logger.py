"""Logs a recorded experiment (an `experiments/results/*.json`-shaped record) to MLflow.

A sibling to `ragas_scorer.py`'s pattern: optional dependency, lazily
imported so importing this module never requires `mlflow` to be
installed. Used by `scripts/record_experiment.py` so every recorded
experiment becomes an MLflow run, not just a JSON/YAML file on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import MLflowConfig

# Config-derived fields in the record schema (MLflow "params"), as opposed
# to measured outcomes ("metrics") -- see _METRIC_FIELDS below.
_PARAM_FIELDS = [
    "generation_model",
    "embedding_model",
    "vector_store",
    "chunking_provider",
    "chunk_size",
    "chunk_overlap",
    "reranker_provider",
    "reranker_model",
    "prompt_id",
    "prompt_version",
    "prompt_file_checksum",
    "retrieval_provider",
    "retrieval_top_k",
    "rerank_top_n",
    "rrf_k",
    "dataset_id",
    "ragas_judge_provider",
    "ragas_judge_model",
]

# Every numeric field in the record schema.
_METRIC_FIELDS = [
    "recall_at_5",
    "recall_at_10",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "mrr",
    "answer_quality",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_correctness",
]


def log_experiment(
    record: dict[str, Any],
    mlflow_config: MLflowConfig,
    artifact_paths: list[Path | None] | None = None,
) -> str | None:
    """Log one `experiments/results/*.json`-shaped record as an MLflow run.

    Fields with a `None` value are skipped rather than logged as
    ``"None"``/``0`` -- MLflow's `log_param`/`log_metric` reject `None`
    outright, and a pre-RAGAS record's `ragas_*` fields simply shouldn't
    appear as metrics on that run, matching how they render as ``"-"`` in
    `scripts/compare_experiments.py`'s table.

    Parameters
    ----------
    record : dict[str, Any]
        A flat record matching the `experiments/results/*.json` schema
        (see `scripts/record_experiment.py`'s `build_experiment_record`).
    mlflow_config : MLflowConfig
        Tracking URI / experiment name / enabled flag.
    artifact_paths : list[Path | None] | None, optional
        Files to attach to the run (e.g. the raw eval-output JSON, the
        recorded record JSON, the config YAML snapshot). `None` entries
        and paths that don't exist are skipped silently, by default None.

    Returns
    -------
    str | None
        The MLflow run_id, or `None` if `mlflow_config.enabled` is False.

    Raises
    ------
    RuntimeError
        If `mlflow_config.enabled` is True but the `mlflow` package isn't
        installed.
    """
    if not mlflow_config.enabled:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "mlflow.enabled is true but the 'mlflow' package isn't installed. "
            "Install it with: pip install .[mlflow]"
        ) from exc

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)

    with mlflow.start_run(run_name=record.get("experiment_id")) as run:
        mlflow.set_tags(
            {
                "experiment_id": record.get("experiment_id") or "",
                "label": record.get("label") or "",
            }
        )
        for field in _PARAM_FIELDS:
            value = record.get(field)
            if value is not None:
                mlflow.log_param(field, value)
        for field in _METRIC_FIELDS:
            value = record.get(field)
            if value is not None:
                mlflow.log_metric(field, float(value))
        for path in artifact_paths or []:
            if path is not None and Path(path).is_file():
                mlflow.log_artifact(str(path))
        return str(run.info.run_id)
