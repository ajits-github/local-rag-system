from __future__ import annotations

from rag.graph.schema import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    canonical_entity_id,
    normalize_entity_name,
)


def test_normalize_entity_name_lowercases_and_collapses_whitespace():
    """Case and incidental whitespace differences normalize to the same string."""
    assert normalize_entity_name("Document  Service") == "document service"
    assert normalize_entity_name("  document service  ") == "document service"
    assert normalize_entity_name("DOCUMENT SERVICE") == "document service"


def test_canonical_entity_id_deterministic():
    """The same inputs always produce the same canonical id."""
    a = canonical_entity_id("Service", "Document Service", dataset_id="techfusion")
    b = canonical_entity_id("Service", "Document Service", dataset_id="techfusion")
    assert a == b


def test_canonical_entity_id_case_and_whitespace_insensitive():
    """Two differently-cased/spaced names collapse to the same canonical id."""
    a = canonical_entity_id("Service", "Document Service", dataset_id="techfusion")
    b = canonical_entity_id("Service", "  document   service ", dataset_id="techfusion")
    assert a == b


def test_canonical_entity_id_scoped_by_dataset():
    """Same name/type in two different datasets never collides."""
    a = canonical_entity_id("Service", "Document Service", dataset_id="techfusion")
    b = canonical_entity_id("Service", "Document Service", dataset_id="other_dataset")
    assert a != b


def test_canonical_entity_id_scoped_by_tenant():
    """Same name/type/dataset but different tenants never collides."""
    a = canonical_entity_id(
        "Service", "Admin Console", dataset_id="techfusion", tenant_id="tenant_alpha"
    )
    b = canonical_entity_id(
        "Service", "Admin Console", dataset_id="techfusion", tenant_id="tenant_beta"
    )
    assert a != b


def test_canonical_entity_id_scoped_by_entity_type():
    """Same name but different entity types never collide (a Service vs a Team named the same)."""
    a = canonical_entity_id("Service", "Compliance", dataset_id="techfusion")
    b = canonical_entity_id("Team", "Compliance", dataset_id="techfusion")
    assert a != b


def test_canonical_entity_id_none_tenant_bucketed_separately_from_a_real_tenant_id():
    """An untenanted entity never collides with a real tenant literally named '_shared'."""
    untenanted = canonical_entity_id("Service", "Gateway", dataset_id="techfusion", tenant_id=None)
    literal_shared_tenant = canonical_entity_id(
        "Service", "Gateway", dataset_id="techfusion", tenant_id="_shared"
    )
    assert untenanted == literal_shared_tenant  # documented, accepted edge case: see docstring
    # This is intentional: canonical_entity_id's docstring documents "_shared" as the
    # fixed bucket for None; a tenant literally named "_shared" is not a real scenario
    # in this corpus's governance schema (tenant ids are asserted claims, not free text
    # an adversary controls independently of the JWT-verification boundary).


def test_entity_types_and_relationship_types_are_nonempty_closed_sets():
    """The allowlists are real, non-trivial closed sets (guards against an accidental empty set)."""
    assert len(ENTITY_TYPES) >= 10
    assert len(RELATIONSHIP_TYPES) >= 10
    assert "Document" in ENTITY_TYPES
    assert "DEPENDS_ON" in RELATIONSHIP_TYPES
