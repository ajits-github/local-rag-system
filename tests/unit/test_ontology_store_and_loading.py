"""Tests: Turtle files parse, the full valid world loads, v1.1 stays additive.

Part of the ontology/semantic-layer milestone (`rag.ontology`); see
`CLAUDE.md`'s "Ontology and semantic layer" section and `docs/ontology.md`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdflib")


def test_ontology_v1_parses_and_is_nonempty(ontology_only_store):
    """Ontology v1 parses and is nonempty."""
    assert len(ontology_only_store) > 50


def test_all_valid_files_load_together(ontology_valid_store):
    """All valid files load together."""
    assert len(ontology_valid_store) > 100


def test_v1_1_loads_additively_without_breaking_v1():
    """V1 1 loads additively without breaking v1."""
    from rag.ontology import paths
    from rag.ontology.store import load_store

    combined = load_store([paths.ONTOLOGY_V1, paths.ONTOLOGY_V1_1])

    q = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    ASK { tech:System owl:deprecated true }
    """
    assert bool(combined.query(q))

    q2 = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    ASK { tech:Vendor a <http://www.w3.org/2002/07/owl#Class> }
    """
    assert bool(combined.query(q2))

    q3 = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    ASK { tech:Service a <http://www.w3.org/2002/07/owl#Class> }
    """
    assert bool(combined.query(q3))


def test_deprecated_system_class_is_equivalent_to_infrastructure_component():
    """Deprecated system class is equivalent to infrastructure component."""
    from rag.ontology import paths
    from rag.ontology.store import load_store

    combined = load_store([paths.ONTOLOGY_V1, paths.ONTOLOGY_V1_1])
    q = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    ASK { tech:System owl:equivalentClass tech:InfrastructureComponent }
    """
    assert bool(combined.query(q))


def test_every_ontology_class_neo4j_label_is_either_a_real_entity_type_or_explicitly_marked_none():
    """Cross-checks every `tech:neo4jLabel` annotation against `rag.graph.schema.ENTITY_TYPES`.

    Keeps the mapping this ontology documents in prose correct if `EntityType` ever changes.
    """
    from rag.graph.schema import ENTITY_TYPES
    from rag.ontology import paths
    from rag.ontology.store import load_store

    store = load_store([paths.ONTOLOGY_V1])
    q = """
    PREFIX tech: <http://techfusion.example/ontology/v1#>
    SELECT ?cls ?label WHERE { ?cls tech:neo4jLabel ?label }
    """
    rows = list(store.query(q))
    assert rows, "expected every class to carry a tech:neo4jLabel annotation"

    for row in rows:
        first_word = str(row.label).split()[0]
        if first_word.lower() == "none":
            continue
        assert first_word in ENTITY_TYPES, (
            f"{row.cls} claims Neo4j label {first_word!r}, which is not in "
            f"rag.graph.schema.ENTITY_TYPES {sorted(ENTITY_TYPES)}"
        )
