from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from rag.config import load_config

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


record_experiment = _load_script("record_experiment")


def _fake_eval_report() -> dict:
    """Build a sample rag.eval.run_eval report with generation metrics."""
    return {
        "generated_at": "2026-08-06T00:00:00+00:00",
        "dataset_id": "unit-test-dataset",
        "retrieval": {"recall@5": 0.9, "recall@10": 0.95},
        "hit_rate": {"hit_rate@5": 0.92, "hit_rate@10": 0.97},
        "mrr": 0.88,
        "latency_ms": {"retrieval_mean": 100.0, "generation_mean": 2000.0, "total_mean": 2100.0},
        "answer_quality": {"mean_overall": 0.5, "mean_answerable": 0.6, "mean_unanswerable": 0.2},
    }


def test_build_experiment_record_reads_config_not_report():
    """Config-derived record fields come from AppConfig, not the report's own summary."""
    # Config values must come from the actual AppConfig, not the report's
    # own (possibly incomplete/stale) embedded summary -- this is what lets
    # older eval reports (missing vector_store/reranker_model) still work.
    config = load_config()
    record = record_experiment.build_experiment_record(
        _fake_eval_report(), config, "experiment_999", "unit test"
    )

    assert record["experiment_id"] == "experiment_999"
    assert record["label"] == "unit test"
    assert record["generation_model"] == config.generation.model_name
    assert record["embedding_model"] == config.embedding.model_name
    assert record["vector_store"] == config.vectorstore.provider
    assert record["chunk_size"] == config.chunking.chunk_size
    assert record["chunk_overlap"] == config.chunking.chunk_overlap
    assert record["reranker_provider"] == config.reranker.provider
    assert record["retrieval_top_k"] == config.retrieval.top_k
    assert record["rerank_top_n"] == config.retrieval.rerank_top_n


def test_build_experiment_record_flattens_metrics_from_report():
    """Retrieval/latency/quality metrics are flattened from the eval report."""
    config = load_config()
    record = record_experiment.build_experiment_record(
        _fake_eval_report(), config, "experiment_999", "unit test"
    )

    assert record["dataset_id"] == "unit-test-dataset"
    assert record["recall_at_5"] == 0.9
    assert record["recall_at_10"] == 0.95
    assert record["hit_rate_at_5"] == 0.92
    assert record["hit_rate_at_10"] == 0.97
    assert record["mrr"] == 0.88
    assert record["answer_quality"] == 0.5
    assert record["retrieval_latency_ms"] == 100.0
    assert record["generation_latency_ms"] == 2000.0
    assert record["total_latency_ms"] == 2100.0


def test_build_experiment_record_handles_missing_generation_fields():
    """A --skip-generation report yields None for generation-derived fields."""
    # A --skip-generation report has no latency_ms/answer_quality at all.
    config = load_config()
    sparse_report = {
        "generated_at": "2026-08-06T00:00:00+00:00",
        "dataset_id": "unit-test-dataset",
        "retrieval": {"recall@5": 0.9, "recall@10": 0.95},
        "hit_rate": {"hit_rate@5": 0.92, "hit_rate@10": 0.97},
        "mrr": 0.88,
    }
    record = record_experiment.build_experiment_record(
        sparse_report, config, "experiment_999", "unit test"
    )

    assert record["answer_quality"] is None
    assert record["retrieval_latency_ms"] is None
    assert record["generation_latency_ms"] is None
    assert record["total_latency_ms"] is None


def test_reranker_model_is_none_for_none_provider():
    """_reranker_model returns None when the reranker provider is 'none'."""
    config = load_config()
    assert config.reranker.provider == "none"
    assert record_experiment._reranker_model(config) is None


def test_build_experiment_record_includes_prompt_fields():
    """The record captures prompt_id/prompt_version and a deterministic file checksum."""
    config = load_config()
    record = record_experiment.build_experiment_record(
        _fake_eval_report(), config, "experiment_999", "unit test"
    )

    expected_checksum = hashlib.sha256(config.prompt_template_path().read_bytes()).hexdigest()
    assert record["prompt_id"] == config.generation.prompt.id
    assert record["prompt_version"] == config.generation.prompt.version
    assert record["prompt_file_checksum"] == expected_checksum
