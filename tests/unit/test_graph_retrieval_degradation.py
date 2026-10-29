"""`GraphRetriever.search` degrades, never raises, when Neo4j is unavailable.

GraphRAG modes are compared on the assumption that a graph outage falls back
to an empty, `degraded=True` result; this pins that path with a fake client.
"""

from __future__ import annotations

from rag.config import load_config
from rag.graph import retrieval as retrieval_module
from rag.graph.client import GraphUnavailableError
from rag.graph.retrieval import GraphRetriever


class _UnavailableClient:
    """A `Neo4jClient` stand-in whose every read fails like a lost connection."""

    def run_read(self, cypher: str, params: dict) -> list[dict]:
        raise GraphUnavailableError("Neo4j unreachable")


def test_search_returns_a_degraded_empty_result_on_neo4j_outage(monkeypatch):
    """No exception escapes; the result is empty, flagged degraded, and counted as a graph error."""
    errors: list[str] = []
    monkeypatch.setattr(retrieval_module.observability_metrics, "observe_error", errors.append)
    retriever = GraphRetriever(load_config(), _UnavailableClient())

    result = retriever.search("What does Document Service depend on?", dataset_id="ds")

    assert result.degraded is True
    assert result.facts == []
    assert result.intent == "unresolved"
    assert errors == ["graph"]
