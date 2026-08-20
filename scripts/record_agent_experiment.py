"""Register a rag.eval.run_agent_eval (+ optional run_agent_ragas_eval) report.

Distinct from scripts/record_experiment.py: the agent report's metric
shape (routing_accuracy/tool_selection_accuracy/... at the top level, no
retrieval@k/hit_rate@k nesting) doesn't fit record_experiment.py's flat
schema, so -- following the same precedent as
scripts/record_attribution_experiment.py -- this writes its own
experiments/results/agentic/<id>.json, a subdirectory
compare_experiments.py's non-recursive glob never sees, leaving the
standard comparison table untouched. It still reuses
rag.eval.mlflow_logger.log_experiment/build_run_name directly (unlike the
attribution recorder): mlflow_logger.py's _PARAM_FIELDS/_METRIC_FIELDS
already carry `agent_enabled` and the 14 `agent_*` metric names from the
Agentic RAG milestone, so no shared logging code needed to change for this
script to work.

`--ragas-output`, if given, merges rag.eval.run_agent_ragas_eval's
aggregate scores into the same record's ragas_* fields (its `judge`
provider/model too) -- optional, since a deterministic-only run has
nothing to merge.

Records every agent-specific prompt's id/version/path/sha256 checksum
(classify/decompose/tool_select/evidence_sufficiency/synthesize) alongside
the classic-path generation prompt's own checksum, so the exact prompt set
behind a given result is always reconstructable later.

Usage:
    python -m rag.eval.run_agent_eval --gold data/eval/agentic_extension_gold.jsonl \
        --dataset-id techfusion --config config/experiments/agentic-rag-baseline-v1.yaml \
        --corpus-version agentic_rag_baseline_v1 --verbose > /tmp/agent_eval_report.json
    python scripts/record_agent_experiment.py \
        --eval-output /tmp/agent_eval_report.json \
        --experiment-id experiment_029 --label agentic_rag_baseline_v1 \
        --config config/experiments/agentic-rag-baseline-v1.yaml
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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results" / "agentic"
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs"

_AGENT_PROMPT_FIELDS = [
    ("classify", "classify_prompt_path"),
    ("decompose", "decompose_prompt_path"),
    ("tool_select", "tool_select_prompt_path"),
    ("evidence_sufficiency", "evidence_sufficiency_prompt_path"),
    ("synthesize", "synthesize_prompt_path"),
]


def _checksum(path: str) -> str:
    """sha256 hex digest of a repo-relative file path's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean_steps(per_example: list[dict[str, Any]]) -> float | None:
    """Mean `steps` across every scored example, or None if there were none."""
    values = [r["steps"] for r in per_example if r.get("steps") is not None]
    return sum(values) / len(values) if values else None


def build_agent_experiment_record(
    agent_report: dict[str, Any],
    ragas_report: dict[str, Any] | None,
    config: AppConfig,
    experiment_id: str,
    label: str,
) -> dict[str, Any]:
    """Flatten an agent eval report (+ optional RAGAS report) into a comparable record.

    Parameters
    ----------
    agent_report : dict[str, Any]
        Parsed JSON output of `rag.eval.run_agent_eval --verbose`.
    ragas_report : dict[str, Any] | None
        Parsed JSON output of `rag.eval.run_agent_ragas_eval`, or None if
        this run is deterministic-only.
    config : AppConfig
        The config that produced `agent_report`.
    experiment_id : str
        Identifier for this experiment, e.g. ``"experiment_029"``.
    label : str
        Short human-readable label, e.g. ``"agentic_rag_baseline_v1"``.

    Returns
    -------
    dict[str, Any]
        Flat record. Fields absent from the agent report's metric shape
        (recall@k, hit_rate@k, and other classic-only fields) are simply
        omitted -- this record is never read by compare_experiments.py's
        standard table.
    """
    corpus_lineage = agent_report.get("corpus_lineage", {})
    per_example = agent_report.get("per_example", [])
    prompt_checksums = {
        f"agent_{name}_prompt_checksum": _checksum(
            str(config.agent_prompt_template_path(getattr(config.agent, field)))
        )
        for name, field in _AGENT_PROMPT_FIELDS
    }
    prompt_paths = {
        f"agent_{name}_prompt_path": getattr(config.agent, field)
        for name, field in _AGENT_PROMPT_FIELDS
    }

    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "label": label,
        "timestamp": agent_report.get("generated_at"),
        "generation_model": config.generation.model_name,
        "embedding_model": config.embedding.model_name,
        "vector_store": config.vectorstore.provider,
        "retrieval_provider": config.retrieval.provider,
        "rrf_k": (config.retrieval.hybrid.rrf_k if config.retrieval.provider == "hybrid" else None),
        "relationship_expansion_enabled": config.retrieval.relationship_expansion.enabled,
        "prompt_id": config.generation.prompt.id,
        "prompt_version": config.generation.prompt.version,
        "prompt_file_checksum": _checksum(str(config.prompt_template_path())),
        "agent_enabled": config.agent.enabled,
        "agent_max_agent_steps": config.agent.max_agent_steps,
        "agent_max_retrieval_attempts": config.agent.max_retrieval_attempts,
        "agent_max_tool_calls": config.agent.max_tool_calls,
        **prompt_paths,
        **prompt_checksums,
        "security_authorization_enabled": config.security.authorization.enabled,
        "security_field_redaction_enabled": config.security.field_redaction.enabled,
        "security_auth_enabled": config.security.auth.enabled,
        "security_rate_limit_enabled": config.security.rate_limit.enabled,
        "security_egress_policy_enabled": config.security.egress_policy.enabled,
        "dataset_id": corpus_lineage.get("dataset_id"),
        "corpus_version": corpus_lineage.get("corpus_version"),
        "corpus_document_count": corpus_lineage.get("document_count"),
        "corpus_chunk_count": corpus_lineage.get("chunk_count"),
        "corpus_image_count": corpus_lineage.get("image_count"),
        "corpus_active_document_count": corpus_lineage.get("active_document_count"),
        "corpus_superseded_document_count": corpus_lineage.get("superseded_document_count"),
        "corpus_tenant_count": corpus_lineage.get("tenant_count"),
        "corpus_gold_record_count": corpus_lineage.get("gold_record_count"),
        "corpus_gold_file_sha256": corpus_lineage.get("gold_file_sha256"),
        "corpus_digest": corpus_lineage.get("corpus_digest"),
        "num_examples": agent_report.get("num_examples"),
        "agent_routing_accuracy": agent_report.get("routing_accuracy", {}).get("rate"),
        "agent_unnecessary_agent_rate": agent_report.get("unnecessary_agent_rate", {}).get("rate"),
        "agent_tool_selection_accuracy": agent_report.get("tool_selection_accuracy", {}).get(
            "rate"
        ),
        "agent_tool_success_rate": agent_report.get("tool_success_rate", {}).get("rate"),
        "agent_average_tool_calls": agent_report.get("average_tool_calls", {}).get("value"),
        "agent_average_agent_steps": _mean_steps(per_example),
        "agent_evidence_sufficiency_accuracy": agent_report.get(
            "evidence_sufficiency_accuracy", {}
        ).get("rate"),
        "agent_retry_success_rate": agent_report.get("retry_success_rate", {}).get("rate"),
        "agent_max_step_termination_rate": agent_report.get("max_step_termination_rate", {}).get(
            "rate"
        ),
        "agent_citation_support_rate": agent_report.get("citation_support_rate", {}).get("rate"),
        "agent_answer_correctness": agent_report.get("agent_answer_correctness", {}).get(
            "mean_score"
        ),
        "agent_latency_ms": agent_report.get("agent_latency_ms", {}).get("overall_mean"),
        "agent_prompt_tokens_mean": agent_report.get("agent_token_usage", {}).get(
            "mean_prompt_tokens"
        ),
        "agent_completion_tokens_mean": agent_report.get("agent_token_usage", {}).get(
            "mean_completion_tokens"
        ),
    }

    if ragas_report is not None:
        ragas_aggregate = ragas_report.get("ragas", {}).get("aggregate", {})
        ragas_judge = ragas_report.get("ragas", {}).get("judge", {})
        record.update(
            {
                "ragas_faithfulness": ragas_aggregate.get("faithfulness"),
                "ragas_answer_relevancy": ragas_aggregate.get("answer_relevancy"),
                "ragas_context_precision": ragas_aggregate.get("context_precision"),
                "ragas_context_recall": ragas_aggregate.get("context_recall"),
                "ragas_answer_correctness": ragas_aggregate.get("answer_correctness"),
                "ragas_noise_sensitivity": ragas_aggregate.get("noise_sensitivity"),
                "ragas_factual_correctness": ragas_aggregate.get("factual_correctness"),
                "ragas_judge_provider": ragas_judge.get("provider"),
                "ragas_judge_model": ragas_judge.get("model_name"),
            }
        )

    return record


def main() -> None:
    """CLI entrypoint: build a record from an agent (+ optional RAGAS) report and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-output", required=True, help="Path to a rag.eval.run_agent_eval JSON report"
    )
    parser.add_argument(
        "--ragas-output",
        default=None,
        help="Optional path to a rag.eval.run_agent_ragas_eval JSON report",
    )
    parser.add_argument("--experiment-id", required=True, help="e.g. experiment_029")
    parser.add_argument(
        "--label", required=True, help="Short human label, e.g. 'agentic_rag_baseline_v1'"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Config file that produced this eval run (snapshotted alongside the record)",
    )
    args = parser.parse_args()

    agent_report = json.loads(Path(args.eval_output).read_text(encoding="utf-8"))
    ragas_report = (
        json.loads(Path(args.ragas_output).read_text(encoding="utf-8"))
        if args.ragas_output
        else None
    )
    config = load_config(args.config)
    record = build_agent_experiment_record(
        agent_report, ragas_report, config, args.experiment_id, args.label
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / f"{args.experiment_id}.json"
    results_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    config_snapshot_path = CONFIGS_DIR / f"{args.experiment_id}.yaml"
    shutil.copy(args.config, config_snapshot_path)

    print(f"Recorded {args.experiment_id} -> {results_path}")

    artifact_paths = [Path(args.eval_output), results_path, config_snapshot_path]
    if args.ragas_output:
        artifact_paths.append(Path(args.ragas_output))
    run_id = log_experiment(record, config.mlflow, artifact_paths=artifact_paths)
    if run_id:
        print(f"Logged to MLflow run {run_id} (tracking_uri={config.mlflow.tracking_uri})")


if __name__ == "__main__":
    main()
