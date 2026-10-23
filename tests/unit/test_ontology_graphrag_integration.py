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
