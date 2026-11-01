"""Dependency questions traverse the direction they ask about (GRAPH-012).

"What does X depend on?" matched the "depends on" cue for "what depends on X",
so it traversed backwards and silently returned no evidence.
"""

from __future__ import annotations

import pytest

from rag.config import load_config
from rag.graph.retrieval import GraphRetriever, _classify_intent


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("What does Document Service depend on?", "dependencies_of"),
        ("Which databases does Billing API rely on?", "dependencies_of"),
        ("What are the dependencies of Document Service?", "dependencies_of"),
        ("What depends on PostgreSQL?", "dependents_of"),
        ("Who depends on PostgreSQL?", "dependents_of"),
        ("Which services rely on PostgreSQL?", "dependents_of"),
        ("What does Document Service do?", "entity_lookup"),
    ],
)
def test_dependency_questions_classify_by_direction(query, intent):
    """Forward and reverse phrasings select opposite traversals."""
    assert _classify_intent(query) == intent


class _FakeClient:
    """Resolves one entity, then returns one dependency row, recording the Cypher."""

    def __init__(self) -> None:
        self.cypher: list[str] = []

    def run_read(self, cypher, params):
        self.cypher.append(cypher)
        if len(self.cypher) == 1:
            node = {"canonical_id": "svc", "name": "Document Service"}
            return [{"e": node, "entity_labels": ["Service"]}]
        dependency = {"canonical_id": "db", "name": "PostgreSQL", "first_seen_document_id": "d1"}
        return [{"dependency": dependency, "dependency_labels": ["Database"], "hops": 1}]


def test_forward_query_traverses_outward_and_states_the_fact_in_that_direction():
    """The fact reads "X depends on Y", from a query walking outward from X."""
    client = _FakeClient()

    result = GraphRetriever(load_config(), client).search(
        "What does Document Service depend on?", dataset_id="ds"
    )

    assert result.intent == "dependencies_of"
    assert "(source)-[:" in client.cypher[1] and "]->(dependency)" in client.cypher[1]
    assert [f.text for f in result.facts] == ["Document Service depends on PostgreSQL"]
    assert [e.name for e in result.facts[0].entities] == ["Document Service", "PostgreSQL"]
