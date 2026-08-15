"""Logs a recorded experiment (an `experiments/results/*.json`-shaped record) to MLflow.

A sibling to `ragas_scorer.py`'s pattern: optional dependency, lazily
imported so importing this module never requires `mlflow` to be
installed. Used by `scripts/record_experiment.py` so every recorded
experiment becomes an MLflow run, not just a JSON/YAML file on disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

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
    "retrieval_candidate_k",
    "reranker_top_n",
    "generation_context_top_n",
    "generation_max_tokens",
    "generation_temperature",
    "generation_seed",
    "rrf_k",
    "dataset_id",
    "ragas_judge_provider",
    "ragas_judge_model",
    # Safety/freshness milestone: dataset-lineage identity (eval/corpus_lineage.py)
    # and the authorization on/off toggle -- all config/identity fields, not
    # measured outcomes, so they're params, not metrics.
    "security_authorization_enabled",
    "corpus_version",
    "corpus_document_count",
    "corpus_chunk_count",
    "corpus_image_count",
    "corpus_active_document_count",
    "corpus_superseded_document_count",
    "corpus_tenant_count",
    "corpus_gold_record_count",
    "corpus_gold_file_sha256",
    "corpus_digest",
]


def _slug(value: Any, default: str = "na") -> str:
    """Lowercase, hyphen-joined, filesystem/URL-safe token from `value`.

    ``None``/empty -> `default` so a missing field never collapses the
    run name into a double-underscore or a leading/trailing separator.
    """
    if value in (None, ""):
        return default
    text = re.sub(r"[^a-zA-Z0-9_]+", "-", str(value)).strip("-").lower()
    return text or default


def build_run_name(record: dict[str, Any]) -> str:
    """Build a human-readable MLflow run name from a record.

    E.g. ``experiment_015_qwen2-5-3b_v2_hybrid_rel-exp``. Never changes
    the MLflow-assigned run UUID (`run.info.run_id`) --
    `run_name` is purely the display label passed to `mlflow.start_run`.
    Built from fields already present in `build_experiment_record`'s
    schema (`scripts/record_experiment.py`), so no new record fields are
    required; a record missing a field (e.g. an older, pre-multimodal
    experiment with no `relationship_expansion_enabled`) just omits that
    segment rather than failing.

    Parameters
    ----------
    record : dict[str, Any]
        A flat `experiments/results/*.json`-shaped record.

    Returns
    -------
    str
        An underscore-joined, MLflow-run-name-safe slug.
    """
    segments = [
        _slug(record.get("experiment_id"), default="run"),
        _slug(record.get("generation_model")),
        _slug(record.get("prompt_version")),
        _slug(record.get("retrieval_provider")),
    ]
    if record.get("relationship_expansion_enabled"):
        segments.append("rel-exp")
    return "_".join(s for s in segments if s != "na")


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
    "ragas_noise_sensitivity",
    "ragas_factual_correctness",
    "prompt_tokens_mean",
    "completion_tokens_mean",
    "unanswerable_refusal_rate",
    "expanded_context_utilization_rate",
    # Safety/freshness milestone: measured outcome rates (see
    # eval/run_eval.py's "safety" report section for exact definitions and
    # lower/higher-is-better direction).
    "safety_unauthorized_retrieval_rate",
    "safety_cross_tenant_leakage_rate",
    "safety_stale_document_error_rate",
    "safety_current_document_retrieval_accuracy",
    "safety_current_document_answer_quality",
    "safety_prompt_injection_success_rate",
    "safety_retrieved_prompt_injection_success_rate",
    "safety_sensitive_data_leakage_rate",
    "safety_refusal_accuracy",
    "safety_false_refusal_rate",
    "safety_poisoned_source_selection_rate",
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

    with mlflow.start_run(run_name=build_run_name(record)) as run:
        mlflow.set_tags(
            {
                "experiment_id": record.get("experiment_id") or "",
                "label": record.get("label") or "",
                "generation_model": record.get("generation_model") or "",
                "prompt_version": record.get("prompt_version") or "",
                "retrieval_provider": record.get("retrieval_provider") or "",
                "reranker_provider": record.get("reranker_provider") or "",
                "relationship_expansion": str(bool(record.get("relationship_expansion_enabled"))),
                "dataset_id": record.get("dataset_id") or "",
                "corpus_version": record.get("corpus_version") or "",
                "security_authorization_enabled": str(
                    bool(record.get("security_authorization_enabled"))
                ),
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
        _log_corpus_input(mlflow, record)
        return str(run.info.run_id)


def _log_corpus_input(mlflow_module: Any, record: dict[str, Any]) -> None:
    """Best-effort `mlflow.log_input` call describing the corpus this run scored.

    Uses `mlflow.data.from_dict` (available on the MLflow versions this
    project pins) to register the corpus-lineage snapshot as a tracked
    "dataset" input on the run, distinct from -- additive to -- the flat
    `corpus_*` params already logged above. Never raises: an older/newer
    MLflow release lacking `mlflow.data`/`from_dict`, or any other failure
    here, is swallowed so a dataset-lineage nicety never breaks the primary
    param/metric logging this function's caller depends on.
    """
    digest = record.get("corpus_digest")
    if not digest:
        return
    try:
        import mlflow.data

        dataset = cast(Any, mlflow.data).from_dict(
            {
                "dataset_id": record.get("dataset_id"),
                "corpus_version": record.get("corpus_version"),
                "document_count": record.get("corpus_document_count"),
                "chunk_count": record.get("corpus_chunk_count"),
                "image_count": record.get("corpus_image_count"),
                "gold_record_count": record.get("corpus_gold_record_count"),
                "gold_file_sha256": record.get("corpus_gold_file_sha256"),
            },
            source=str(record.get("dataset_id") or "unknown"),
            name=digest,
        )
        mlflow_module.log_input(dataset, context="eval")
    except Exception:  # noqa: BLE001 -- best-effort, see docstring
        pass
