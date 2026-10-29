"""Unit tests for `GraphRAGPipeline.answer()`'s `graph_strategy_executed` field.

Regression for EVAL-013: `GraphRAGPipeline.answer()` used to silently
substitute plain vector retrieval whenever `graph.enabled=False` (or no
retriever could be built), with nothing in the returned dict indicating
the substitution happened. A graph-vs-vector comparison run against the
shipped, graph-disabled default could therefore report several
differently-labeled strategies that all executed identically.
"""

from __future__ import annotations

import pytest

from rag.config import load_config
from rag.eval.run_graph_eval import compare_strategies
from rag.graph.pipeline import GraphRAGPipeline


class _FakeRetrievalPipeline:
    """Only implements what the graph-disabled fallback path in GraphRAGPipeline.answer() calls."""

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        return {"answer": "fake answer", "sources": []}


def _config(*, graph_enabled: bool):
    config = load_config().model_copy(deep=True)
    config.graph.enabled = graph_enabled
    return config


def test_graph_disabled_reports_vector_only_graph_disabled_for_a_non_vector_strategy():
    """Requesting graph_only with graph.enabled=False is reported honestly, not as graph_only."""
    pipeline = GraphRAGPipeline(_config(graph_enabled=False), _FakeRetrievalPipeline())

    result = pipeline.answer("q", strategy_override="graph_only")

    assert result["graph_strategy_executed"] == "vector_only_graph_disabled"


def test_graph_disabled_vector_only_request_is_reported_as_vector_only():
    """Explicitly requesting vector_only is never mislabeled as a degrade."""
    pipeline = GraphRAGPipeline(_config(graph_enabled=False), _FakeRetrievalPipeline())

    result = pipeline.answer("q", strategy_override="vector_only")

    assert result["graph_strategy_executed"] == "vector_only"


def test_compare_strategies_raises_instead_of_silently_running_everything_vector_only(tmp_path):
    """compare_strategies must fail loud, not produce a mislabeled comparison table."""
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        '{"question": "q", "relevant_documents": []}\n',
        encoding="utf-8",
    )
    config = _config(graph_enabled=False)

    with pytest.raises(ValueError, match="graph.enabled is False"):
        compare_strategies(
            str(gold_path),
            "test-dataset",
            ["vector_only", "graph_only", "vector_plus_graph"],
            config=config,
        )
