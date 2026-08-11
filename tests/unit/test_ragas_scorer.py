from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from rag.embedders.base import Embedder
from rag.eval import ragas_scorer
from rag.eval.ragas_scorer import build_dataset, score
from rag.generation.base import LLM


class FakeLLM(LLM):
    """Minimal LLM double satisfying the `LLM` ABC for adapter construction."""

    def generate(self, prompt: str) -> str:
        """Return a fixed placeholder string."""
        return "fake"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder(Embedder):
    """Minimal Embedder double satisfying the `Embedder` ABC for adapter construction."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


class _FakeDataset(dict):
    """Stand-in for `datasets.Dataset` -- just a dict of columns."""

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> _FakeDataset:
        """Mirror `Dataset.from_dict`'s constructor."""
        return cls(data)


class _FakeColumn(list):
    """A dataframe column double supporting `.mean()`."""

    def mean(self) -> float:
        """Arithmetic mean of the column's values."""
        return sum(self) / len(self)


class _FakeRow:
    """One `.iloc[i]` row double supporting `row[metric_name]`."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Store this row's column values."""
        self._data = data

    def __getitem__(self, key: str) -> Any:
        """Return the value for `key`."""
        return self._data[key]


class _FakeIloc:
    """Stand-in for `DataFrame.iloc`."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store the underlying rows."""
        self._rows = rows

    def __getitem__(self, i: int) -> _FakeRow:
        """Return row `i` as a `_FakeRow`."""
        return _FakeRow(self._rows[i])


class _FakeDataFrame:
    """Minimal `pandas.DataFrame` double supporting what `ragas_scorer.score` needs."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Build `.columns`/`.iloc` from `rows`."""
        self._rows = rows
        self.columns: set[str] = set()
        for row in rows:
            self.columns |= set(row.keys())
        self.iloc = _FakeIloc(rows)

    def __getitem__(self, column: str) -> _FakeColumn:
        """Return `column` as a `_FakeColumn`."""
        return _FakeColumn(row[column] for row in self._rows)


def _install_fake_ragas(monkeypatch, *, metric_names: list[str], per_row_scores: dict[str, float]):
    """Inject fake `ragas`/`ragas.metrics`/`ragas.llms`/`ragas.embeddings`/`datasets` modules."""
    metrics_module = types.SimpleNamespace(**dict.fromkeys(metric_names, object()))
    llms_module = types.SimpleNamespace(LangchainLLMWrapper=lambda x: x)
    embeddings_module = types.SimpleNamespace(LangchainEmbeddingsWrapper=lambda x: x)

    def fake_evaluate(dataset, metrics, llm, embeddings):
        n = len(dataset["user_input"])
        rows = [dict(per_row_scores) for _ in range(n)]
        return types.SimpleNamespace(to_pandas=lambda: _FakeDataFrame(rows))

    ragas_module = types.SimpleNamespace(
        evaluate=fake_evaluate,
        metrics=metrics_module,
        llms=llms_module,
        embeddings=embeddings_module,
    )
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_module)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=_FakeDataset))


def _rows() -> list[dict[str, Any]]:
    """Two sample scoring rows."""
    return [
        {
            "question_index": 0,
            "question": "What is pgvector?",
            "unanswerable": False,
            "answer": "A Postgres extension.",
            "contexts": ["pgvector is a Postgres extension for vector similarity search."],
            "expected_answer": "A Postgres extension for vector search.",
        },
        {
            "question_index": 1,
            "question": "What is the parental leave policy?",
            "unanswerable": True,
            "answer": "I don't know.",
            "contexts": ["No relevant policy found."],
            "expected_answer": "The knowledge base does not contain this policy.",
        },
    ]


def test_score_missing_ragas_package_raises_runtime_error(monkeypatch):
    """score() raises RuntimeError pointing at the ragas extra when ragas isn't importable.

    `sys.modules["ragas"] = None` forces the next `import ragas` to raise
    `ImportError` regardless of whether the `ragas` extra actually happens
    to be installed in the active venv (it is, in this project's own dev
    environment, since real RAGAS runs are part of this repo's workflow --
    see CLAUDE.md) -- this simulates the "not installed" branch
    deterministically rather than depending on venv state.
    """
    monkeypatch.setitem(sys.modules, "ragas", None)
    with pytest.raises(RuntimeError, match=r"pip install \.\[ragas\]"):
        score([], judge_llm=FakeLLM(), embedder=FakeEmbedder())


def test_build_dataset_missing_datasets_package_raises_runtime_error(monkeypatch):
    """build_dataset() raises RuntimeError pointing at the ragas extra when datasets is missing.

    See `test_score_missing_ragas_package_raises_runtime_error`'s
    docstring for why `sys.modules` is patched directly rather than
    relying on the package actually being absent.
    """
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(RuntimeError, match=r"pip install \.\[ragas\]"):
        build_dataset(_rows())


def test_score_builds_aggregate_and_per_question_from_fake_evaluate_result(monkeypatch):
    """score() extracts per-question scores and aggregate means from ragas.evaluate()'s result."""
    _install_fake_ragas(
        monkeypatch,
        metric_names=ragas_scorer.METRIC_NAMES,
        per_row_scores={
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
            "answer_correctness": 0.5,
        },
    )

    result = score(_rows(), judge_llm=FakeLLM(), embedder=FakeEmbedder())

    assert set(result["metrics_used"]) == set(ragas_scorer.METRIC_NAMES)
    assert result["metrics_failed"] == {}
    assert result["aggregate"]["faithfulness"] == 0.9
    assert len(result["per_question"]) == 2
    assert result["per_question"][0]["scores"]["faithfulness"] == 0.9
    assert result["per_question"][1]["unanswerable"] is True
    assert result["caveat"] == ragas_scorer.CAVEAT


def test_score_metrics_failed_when_one_metric_import_raises(monkeypatch):
    """One metric missing from the installed ragas version doesn't block the others."""
    available = [m for m in ragas_scorer.METRIC_NAMES if m != "answer_correctness"]
    _install_fake_ragas(
        monkeypatch,
        metric_names=available,
        per_row_scores={"faithfulness": 0.9, "answer_relevancy": 0.8},
    )

    result = score(_rows(), judge_llm=FakeLLM(), embedder=FakeEmbedder())

    assert "answer_correctness" in result["metrics_failed"]
    assert "answer_correctness" not in result["metrics_used"]
    assert set(result["metrics_used"]) == set(available)


def test_score_empty_rows_returns_empty_report(monkeypatch):
    """score() with no rows returns an empty report without calling ragas.evaluate()."""
    _install_fake_ragas(
        monkeypatch, metric_names=ragas_scorer.METRIC_NAMES, per_row_scores={"faithfulness": 0.9}
    )

    result = score([], judge_llm=FakeLLM(), embedder=FakeEmbedder())

    assert result["aggregate"] == {}
    assert result["per_question"] == []
