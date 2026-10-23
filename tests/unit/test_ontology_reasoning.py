"""Tests: RDFS/OWL-RL inference, explicit-vs-inferred, disjointness checking."""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")

TECH = "http://techfusion.example/ontology/v1#"
INST = "http://techfusion.example/graph/techfusion_graph_demo/"


def test_rdfs_subclass_entailment_types_postgresql_as_service(ontology_valid_store):
    """Rdfs subclass entailment types postgresql as service."""
    from rag.ontology.reasoning import materialize_rdfs

    before_count = len(ontology_valid_store)
    report = materialize_rdfs(ontology_valid_store)
    assert report.explicit_triple_count == before_count
    assert report.total_triple_count > before_count
    assert report.inferred_triple_count > 0

    ask = f"""
    PREFIX tech: <{TECH}>
    ASK {{ <{INST}database-postgresql> a tech:Service }}
    """
    assert bool(ontology_valid_store.query(ask))


def test_rdfs_does_not_mutate_a_copy_taken_before_it_ran(ontology_valid_store):
    """Rdfs does not mutate a copy taken before it ran."""
    from rag.ontology.reasoning import materialize_rdfs

    before = ontology_valid_store.copy()
    materialize_rdfs(ontology_valid_store)
    assert len(before) < len(ontology_valid_store)


def test_inferred_only_triples_reports_the_diff(ontology_valid_store):
    """Inferred only triples reports the diff."""
    from rag.ontology.reasoning import inferred_only_triples, materialize_rdfs

    before = ontology_valid_store.copy()
    materialize_rdfs(ontology_valid_store)
    new_triples = inferred_only_triples(before, ontology_valid_store)
    assert len(new_triples) == len(ontology_valid_store) - len(before)
    assert all(t not in set(before.graph) for t in new_triples)


def test_owl_rl_functional_property_entails_same_as(ontology_reasoning_fixtures_store):
    """Owl rl functional property entails same as."""
    from rag.ontology.reasoning import materialize_owl_rl

    materialize_owl_rl(ontology_reasoning_fixtures_store)
    q = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    ASK {{ <{INST}team-billing-ops> owl:sameAs <{INST}team-finance-platform> }}
    """
    assert bool(ontology_reasoning_fixtures_store.query(q))


def test_disjoint_violation_detected_in_fixture(ontology_reasoning_fixtures_store):
    """Disjoint violation detected in fixture."""
    from rag.ontology.reasoning import find_disjoint_class_violations, materialize_owl_rl

    materialize_owl_rl(ontology_reasoning_fixtures_store)
    violations = find_disjoint_class_violations(ontology_reasoning_fixtures_store)
    flagged = {v[0] for v in violations}
    assert f"{INST}service-dual-typed-anomaly" in flagged


def test_no_disjoint_violations_on_valid_data(ontology_valid_store):
    """No disjoint violations on valid data."""
    from rag.ontology.reasoning import find_disjoint_class_violations, materialize_owl_rl

    materialize_owl_rl(ontology_valid_store)
    assert find_disjoint_class_violations(ontology_valid_store) == []


def test_owl_rl_inverse_of_entails_isdependencyof(ontology_valid_store):
    """Owl rl inverse of entails isdependencyof."""
    from rag.ontology.reasoning import materialize_owl_rl

    raw_ask = f"""
    PREFIX tech: <{TECH}>
    ASK {{ <{INST}database-postgresql> tech:isDependencyOf ?x }}
    """
    assert not bool(ontology_valid_store.query(raw_ask))

    materialize_owl_rl(ontology_valid_store)
    assert bool(ontology_valid_store.query(raw_ask))


def test_domain_axiom_is_an_inference_trigger_not_a_validation_constraint():
    """Isolated reproduction of the mechanism behind KNOWN_ISSUES.md's ONT-001 entry.

    Also documented on `tech:dependsOn`'s own comment in
    `data/ontology/schema/techfusion_ontology_v1.ttl`.
    """
    from rag.ontology.reasoning import materialize_rdfs
    from rag.ontology.store import RdflibStore

    toy = """
    @prefix ex: <http://example.org/toy#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    ex:worksAt a rdf:Property ;
        rdfs:domain ex:Person .

    ex:ACMECorp ex:worksAt ex:SomeVendor .
    """
    store = RdflibStore()
    store.graph.parse(data=toy, format="turtle")

    ask = "PREFIX ex: <http://example.org/toy#> ASK { ex:ACMECorp a ex:Person }"
    assert not bool(store.query(ask))

    materialize_rdfs(store)
    assert bool(store.query(ask))
