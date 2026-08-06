from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config import MLflowConfig
from rag.eval.mlflow_logger import log_experiment


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
    assert fake.run_names == ["experiment_042"]


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


def test_log_experiment_returns_run_id(monkeypatch: pytest.MonkeyPatch):
    """log_experiment returns the MLflow run's run_id on success."""
    _install_fake_mlflow(monkeypatch)
    config = MLflowConfig()

    run_id = log_experiment(_record(), config)

    assert run_id == "fake-run-id"
