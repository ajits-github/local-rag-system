"""Tests: bounded ontology-aware concept expansion."""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")

TECH = "http://techfusion.example/ontology/v1#"


def test_service_expands_to_its_three_subclasses(ontology_only_store):
    """Service expands to its three subclasses."""
    from rag.ontology.query_expansion import expand_concept

    result = expand_concept(ontology_only_store, TECH + "Service", max_depth=2, max_results=20)
    short = {c.rsplit("#", 1)[-1] for c in result.expanded_classes}
    assert short == {"APIService", "DatabaseService", "MessagingService"}
    assert result.hit_depth_cap is False
    assert result.hit_result_cap is False


def test_leaf_class_expands_to_nothing(ontology_only_store):
    """Leaf class expands to nothing."""
    from rag.ontology.query_expansion import expand_concept

    result = expand_concept(
        ontology_only_store, TECH + "DatabaseService", max_depth=2, max_results=20
    )
    assert result.expanded_classes == []


def _build_chain_store(length: int):
    from rdflib import RDFS, Namespace

    from rag.ontology.store import RdflibStore

    ns = Namespace("http://example.org/chain#")
    store = RdflibStore()
    prev = ns.Root
    for i in range(length):
        current = ns[f"C{i}"]
        store.graph.add((current, RDFS.subClassOf, prev))
        prev = current
    return store


def test_depth_cap_stops_expansion_even_with_more_subclasses_available():
    """Depth cap stops expansion even with more subclasses available."""
    from rag.ontology.query_expansion import expand_concept

    chain = _build_chain_store(5)
    result = expand_concept(chain, "http://example.org/chain#Root", max_depth=2, max_results=20)
    assert len(result.expanded_classes) == 2
    assert result.hit_depth_cap is True
    assert result.hit_result_cap is False


def test_result_cap_stops_expansion_before_depth_cap():
    """Result cap stops expansion before depth cap."""
    from rag.ontology.query_expansion import expand_concept

    chain = _build_chain_store(5)
    result = expand_concept(chain, "http://example.org/chain#Root", max_depth=10, max_results=2)
    assert len(result.expanded_classes) == 2
    assert result.hit_result_cap is True


def test_expansion_never_includes_the_root_class_itself(ontology_only_store):
    """Expansion never includes the root class itself."""
    from rag.ontology.query_expansion import expand_concept

    result = expand_concept(ontology_only_store, TECH + "Service", max_depth=2, max_results=20)
    assert (TECH + "Service") not in result.expanded_classes
