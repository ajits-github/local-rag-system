from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config import MLflowConfig
from rag.eval.mlflow_logger import build_run_name, log_experiment


class _FakeRunInfo:
    """Stand-in for `mlflow.ActiveRun.info`, exposing just `run_id`."""

    def __init__(self, run_id: str) -> None:
        """Store the fake run id."""
        self.run_id = run_id


class _FakeActiveRun:
    """Stand-in for the context manager `mlflow.start_run()` returns."""

    def __init__(self, run_id: str) -> None:
        """Wrap `run_id` in a fake `.info`."""
        self.info = _FakeRunInfo(run_id)

    def __enter__(self) -> _FakeActiveRun:
        """Return self, mirroring a real MLflow run context manager."""
        return self

    def __exit__(self, *exc: object) -> bool:
        """Never suppress exceptions."""
        return False


class _FakeMlflow:
    """Stand-in for the `mlflow` module: records every call for assertions."""

    def __init__(self) -> None:
        """Start with empty call-tracking state."""
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.tags: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []
        self.run_names: list[str | None] = []

    def set_tracking_uri(self, uri: str) -> None:
        """Record the tracking URI passed."""
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        """Record the experiment name passed."""
        self.experiment_name = name

    def start_run(self, run_name: str | None = None) -> _FakeActiveRun:
        """Record `run_name` and return a fake active run."""
        self.run_names.append(run_name)
        return _FakeActiveRun("fake-run-id")

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Merge `tags` into the recorded tag dict."""
        self.tags.update(tags)

    def log_param(self, key: str, value: Any) -> None:
        """Record one logged param."""
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        """Record one logged metric."""
        self.metrics[key] = value

    def log_artifact(self, path: str) -> None:
        """Record one logged artifact path."""
        self.artifacts.append(path)


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflow:
    """Inject a fake `mlflow` module into sys.modules for the duration of a test."""
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _record(**overrides: Any) -> dict[str, Any]:
    """Build a minimal experiments/results/*.json-shaped record."""
    base = {
        "experiment_id": "experiment_999",
        "label": "unit test",
        "generation_model": "qwen2.5:1.5b",
        "embedding_model": "all-MiniLM-L6-v2",
        "recall_at_5": 0.9,
        "mrr": 0.85,
        "ragas_faithfulness": None,
        "reranker_model": None,
    }
    base.update(overrides)
    return base


def test_log_experiment_returns_none_and_skips_entirely_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """log_experiment is a no-op (no mlflow calls at all) when mlflow.enabled is False."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig(enabled=False)

    run_id = log_experiment(_record(), config)

    assert run_id is None
    assert fake.run_names == []


def test_log_experiment_missing_package_raises_runtime_error(monkeypatch: pytest.MonkeyPatch):
    """log_experiment raises RuntimeError pointing at the mlflow extra when it's missing."""
    monkeypatch.setitem(sys.modules, "mlflow", None)  # forces `import mlflow` to raise ImportError
    config = MLflowConfig(enabled=True)

    with pytest.raises(RuntimeError, match=r"pip install \.\[mlflow\]"):
        log_experiment(_record(), config)


def test_log_experiment_sets_tracking_uri_and_experiment_name(monkeypatch: pytest.MonkeyPatch):
    """log_experiment configures MLflow's tracking URI and experiment name from config."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig(tracking_uri="file:./mlruns", experiment_name="my-exp")

    log_experiment(_record(), config)

    assert fake.tracking_uri == "file:./mlruns"
    assert fake.experiment_name == "my-exp"


def test_log_experiment_logs_params_and_metrics(monkeypatch: pytest.MonkeyPatch):
    """Config-derived fields become params; numeric outcome fields become metrics."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(generation_model="qwen2.5:1.5b", recall_at_5=0.9), config)

    assert fake.params["generation_model"] == "qwen2.5:1.5b"
    assert fake.metrics["recall_at_5"] == 0.9


def test_log_experiment_logs_retrieval_provider_and_rrf_k(monkeypatch: pytest.MonkeyPatch):
    """retrieval_provider/rrf_k are logged as params, not silently dropped."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(retrieval_provider="hybrid", rrf_k=60), config)

    assert fake.params["retrieval_provider"] == "hybrid"
    assert fake.params["rrf_k"] == 60


def test_log_experiment_skips_none_valued_fields(monkeypatch: pytest.MonkeyPatch):
    """A None-valued field (e.g. a pre-RAGAS record's ragas_* fields) is never logged."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(ragas_faithfulness=None, reranker_model=None), config)

    assert "ragas_faithfulness" not in fake.metrics
    assert "reranker_model" not in fake.params


def test_log_experiment_tags_experiment_id_and_label(monkeypatch: pytest.MonkeyPatch):
    """The run is tagged with this project's own experiment_id/label."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(experiment_id="experiment_042", label="my label"), config)

    assert fake.tags["experiment_id"] == "experiment_042"
    assert fake.tags["label"] == "my label"


def test_log_experiment_uses_readable_run_name(monkeypatch: pytest.MonkeyPatch):
    """The MLflow run_name is a readable slug, not the bare experiment_id."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()
    record = _record(
        experiment_id="experiment_015",
        generation_model="qwen2.5:3b",
        prompt_version="v2",
        retrieval_provider="hybrid",
        relationship_expansion_enabled=True,
    )

    log_experiment(record, config)

    assert fake.run_names == [build_run_name(record)]
    assert fake.run_names[0] != "experiment_015"
    assert "experiment_015" in fake.run_names[0]
    assert "rel-exp" in fake.run_names[0]


def test_log_experiment_sets_new_provenance_tags(monkeypatch: pytest.MonkeyPatch):
    """generation_model/prompt_version/retrieval/reranker/relationship/dataset are tagged."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()
    record = _record(
        generation_model="qwen2.5:3b",
        prompt_version="v2",
        retrieval_provider="hybrid",
        reranker_provider="none",
        relationship_expansion_enabled=True,
        dataset_id="techfusion",
    )

    log_experiment(record, config)

    assert fake.tags["generation_model"] == "qwen2.5:3b"
    assert fake.tags["prompt_version"] == "v2"
    assert fake.tags["retrieval_provider"] == "hybrid"
    assert fake.tags["reranker_provider"] == "none"
    assert fake.tags["relationship_expansion"] == "True"
    assert fake.tags["dataset_id"] == "techfusion"


def test_log_experiment_provenance_tags_default_safely_for_older_records(
    monkeypatch: pytest.MonkeyPatch,
):
    """A pre-multimodal record (no relationship_expansion_enabled/dataset_id) tags cleanly."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(), config)  # no relationship_expansion_enabled/dataset_id override

    assert fake.tags["relationship_expansion"] == "False"
    assert fake.tags["dataset_id"] == ""


def test_build_run_name_omits_missing_segments():
    """A record missing optional fields still produces a valid, readable run name."""
    name = build_run_name({"experiment_id": "experiment_001"})
    assert name == "experiment_001"


def test_build_run_name_includes_rel_exp_only_when_enabled():
    """The 'rel-exp' segment appears only when relationship_expansion_enabled is truthy."""
    base = {
        "experiment_id": "experiment_011",
        "generation_model": "qwen2.5:1.5b",
        "prompt_version": "v2",
        "retrieval_provider": "hybrid",
    }
    without = build_run_name(base)
    with_expansion = build_run_name({**base, "relationship_expansion_enabled": True})

    assert "rel-exp" not in without
    assert with_expansion == f"{without}_rel-exp"


def test_build_run_name_includes_agentic_only_when_enabled():
    """The 'agentic' segment appears only when agent_enabled is truthy."""
    base = {
        "experiment_id": "experiment_030",
        "generation_model": "qwen2.5:3b",
        "prompt_version": "v3",
        "retrieval_provider": "hybrid",
    }
    without = build_run_name(base)
    with_agent = build_run_name({**base, "agent_enabled": True})

    assert "agentic" not in without
    assert with_agent == f"{without}_agentic"


def test_log_experiment_attaches_existing_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Only artifact paths that actually exist on disk get logged; None entries are skipped."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()
    real_file = tmp_path / "record.json"
    real_file.write_text("{}", encoding="utf-8")
    missing_file = tmp_path / "does-not-exist.json"

    log_experiment(_record(), config, artifact_paths=[real_file, missing_file, None])

    assert str(real_file) in fake.artifacts
    assert str(missing_file) not in fake.artifacts
    assert len(fake.artifacts) == 1


def test_log_experiment_logs_generation_context_latency_milestone_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    """generation_max_tokens (param) and the new token/refusal/expansion metrics are logged."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()
    record = _record(
        generation_max_tokens=256,
        prompt_tokens_mean=812.5,
        completion_tokens_mean=64.0,
        unanswerable_refusal_rate=0.75,
        expanded_context_utilization_rate=0.4,
        ragas_noise_sensitivity=0.12,
        ragas_factual_correctness=0.66,
    )

    log_experiment(record, config)

    assert fake.params["generation_max_tokens"] == 256
    assert fake.metrics["prompt_tokens_mean"] == 812.5
    assert fake.metrics["completion_tokens_mean"] == 64.0
    assert fake.metrics["unanswerable_refusal_rate"] == 0.75
    assert fake.metrics["expanded_context_utilization_rate"] == 0.4
    assert fake.metrics["ragas_noise_sensitivity"] == 0.12
    assert fake.metrics["ragas_factual_correctness"] == 0.66


def test_log_experiment_logs_generation_temperature_and_seed_params(
    monkeypatch: pytest.MonkeyPatch,
):
    """generation_temperature/generation_seed are logged as params (config, not metrics)."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(generation_temperature=0.0, generation_seed=42), config)

    assert fake.params["generation_temperature"] == 0.0
    assert fake.params["generation_seed"] == 42


def test_log_experiment_omits_generation_seed_param_when_none(monkeypatch: pytest.MonkeyPatch):
    """A None generation_seed (non-deterministic run) is never logged as a param."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(generation_seed=None), config)

    assert "generation_seed" not in fake.params


def test_log_experiment_logs_corpus_lineage_and_safety_fields(monkeypatch: pytest.MonkeyPatch):
    """Corpus-lineage identity fields are params; safety outcome rates are metrics."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()
    record = _record(
        security_authorization_enabled=True,
        security_field_redaction_enabled=True,
        corpus_version="2026-08-14-safety-v1",
        corpus_document_count=18,
        corpus_chunk_count=210,
        corpus_image_count=4,
        corpus_digest="abc123",
        safety_document_unauthorized_retrieval_rate=0.0,
        safety_false_refusal_rate=0.05,
        safety_sensitive_data_false_redaction_rate=0.0,
    )

    log_experiment(record, config)

    assert fake.params["corpus_version"] == "2026-08-14-safety-v1"
    assert fake.params["corpus_document_count"] == 18
    assert fake.params["corpus_digest"] == "abc123"
    assert fake.tags["corpus_version"] == "2026-08-14-safety-v1"
    assert fake.tags["security_authorization_enabled"] == "True"
    assert fake.tags["security_field_redaction_enabled"] == "True"
    assert fake.metrics["safety_document_unauthorized_retrieval_rate"] == 0.0
    assert fake.metrics["safety_false_refusal_rate"] == 0.05
    assert fake.metrics["safety_sensitive_data_false_redaction_rate"] == 0.0


def test_log_experiment_omits_safety_metrics_when_none(monkeypatch: pytest.MonkeyPatch):
    """A pre-safety-milestone record (no safety_* fields) never logs those metrics."""
    fake = _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    log_experiment(_record(), config)

    assert "safety_document_unauthorized_retrieval_rate" not in fake.metrics
    assert "safety_sensitive_data_false_redaction_rate" not in fake.metrics
    assert "corpus_digest" not in fake.params


def test_log_experiment_returns_run_id(monkeypatch: pytest.MonkeyPatch):
    """log_experiment returns the MLflow run's run_id on success."""
    _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    run_id = log_experiment(_record(), config)

    assert run_id == "fake-run-id"
