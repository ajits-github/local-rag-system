"""Tests: naive vs. ontology-aware type-based retrieval (docs/ontology.md sections 12, 13, 18)."""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")

TECH = "http://techfusion.example/ontology/v1#"


def _short(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            iri = iri.rsplit(sep, 1)[-1]
    return iri


def test_naive_match_misses_subclassed_infrastructure(ontology_valid_store):
    """Naive match misses subclassed infrastructure."""
    from rag.ontology.graphrag_integration import naive_type_match

    result = naive_type_match(ontology_valid_store, TECH + "Service")
    names = {_short(e) for e in result.entities}
    assert names == {
        "service-document-service",
        "service-workflow-engine",
        "service-notification-service",
        "service-rag-service",
        "service-gateway",
        "service-compliance-ledger",
    }


def test_ontology_aware_match_includes_subclassed_infrastructure(ontology_valid_store):
    """Ontology aware match includes subclassed infrastructure."""
    from rag.ontology.graphrag_integration import ontology_aware_type_match

    result = ontology_aware_type_match(ontology_valid_store, ontology_valid_store, TECH + "Service")
    names = {_short(e) for e in result.entities}
    assert "database-postgresql" in names
    assert "database-opensearch" in names
    assert "technology-kafka" in names
    assert len(names) == 10


def test_ontology_aware_match_matches_naive_when_class_has_no_subclasses(ontology_valid_store):
    """Ontology aware match matches naive when class has no subclasses."""
    from rag.ontology.graphrag_integration import naive_type_match, ontology_aware_type_match

    naive = naive_type_match(ontology_valid_store, TECH + "DatabaseService")
    aware = ontology_aware_type_match(
        ontology_valid_store, ontology_valid_store, TECH + "DatabaseService"
    )
    assert set(naive.entities) == set(aware.entities)
    assert aware.expanded_classes == []


def test_unanswerable_concept_returns_empty_both_ways(ontology_valid_store):
    """Unanswerable concept returns empty both ways."""
    from rag.ontology.graphrag_integration import naive_type_match, ontology_aware_type_match

    bogus_class = TECH + "ServiceLevelAgreement"
    aware_result = ontology_aware_type_match(
        ontology_valid_store, ontology_valid_store, bogus_class
    )
    assert naive_type_match(ontology_valid_store, bogus_class).entities == []
    assert aware_result.entities == []


def test_resolve_neo4j_labels_skips_entries_with_no_real_label(ontology_only_store):
    """`_resolve_neo4j_labels` must skip any class whose `tech:neo4jLabel` starts with 'none'.

    Used by `expand_and_query_live_neo4j`, the one function in this
    package that talks to a live Neo4j instance; not otherwise covered
    by tests/unit, since reaching an unreachable real network port is
    not a unit test (manually verified instead).
    """
    from rag.ontology.graphrag_integration import _resolve_neo4j_labels

    labels = _resolve_neo4j_labels(
        ontology_only_store,
        [TECH + "Service", TECH + "InfrastructureComponent"],
    )
    assert "Service" in labels
    assert "none" not in [label.lower() for label in labels]


class _FakeNode(dict):
    """A Neo4j `Node` stand-in: a property dict."""


class _RecordingNeo4jClient:
    """Fake `Neo4jClient` returning one node per tenant and recording the Cypher it ran."""

    instances: list[_RecordingNeo4jClient] = []

    def __init__(self, config) -> None:
        self.cypher: list[str] = []
        _RecordingNeo4jClient.instances.append(self)

    def health_check(self) -> bool:
        return True

    def run_read(self, cypher, params):
        self.cypher.append(cypher)
        return [
            {"node": _FakeNode(name="Alpha API", tenant_id="tenant_alpha"), "labels": ["Service"]},
            {"node": _FakeNode(name="Beta API", tenant_id="tenant_beta"), "labels": ["Service"]},
        ]

    def close(self) -> None:
        pass


def test_live_neo4j_helper_filters_unauthorized_entities_and_orders_results(
    ontology_only_store, monkeypatch
):
    """GRAPH-009: the live helper applies the graph authorization predicate and re-check.

    A row the Cypher predicate let through (here, the fake returns both
    tenants regardless) is still dropped by the Python re-check, and the
    query orders before `LIMIT` so results are reproducible.
    """
    from rag.graph import client as client_module
    from rag.ontology.graphrag_integration import expand_and_query_live_neo4j
    from rag.retrieval.authorization import AuthorizationContext

    _RecordingNeo4jClient.instances.clear()
    monkeypatch.setattr(client_module, "Neo4jClient", _RecordingNeo4jClient)

    rows = expand_and_query_live_neo4j(
        ontology_only_store,
        TECH + "Service",
        dataset_id="techfusion_graph_demo",
        auth=AuthorizationContext(tenant_id="tenant_alpha", roles=[]),
    )

    assert rows == [{"name": "Alpha API", "labels": ["Service"]}]
    cypher = _RecordingNeo4jClient.instances[0].cypher[0]
    assert "ORDER BY n.name, n.canonical_id" in cypher
    assert "tenant_id" in cypher
