"""Tests: SHACL validation, raw vs. RDFS-closed, valid vs. deliberately invalid data."""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")


def test_valid_data_has_exactly_the_known_owner_gap(ontology_valid_store, ontology_shapes_store):
    """Valid data has exactly the known owner gap."""
    from rag.ontology.validation import run_shacl_validation

    report = run_shacl_validation(ontology_valid_store, ontology_shapes_store, inference=None)
    assert report.conforms is False
    assert report.violation_count == 5
    assert all("owning Team" in s for s in report.violation_summaries)


def test_valid_data_raw_and_rdfs_closed_agree(ontology_valid_store, ontology_shapes_store):
    """Valid data raw and rdfs closed agree."""
    from rag.ontology.validation import run_shacl_validation

    raw = run_shacl_validation(ontology_valid_store, ontology_shapes_store, inference=None)
    rdfs_closed = run_shacl_validation(
        ontology_valid_store, ontology_shapes_store, inference="rdfs"
    )
    assert raw.violation_count == rdfs_closed.violation_count
    assert set(raw.violation_summaries) == set(rdfs_closed.violation_summaries)


def test_subclass_typed_individual_is_caught_by_superclass_shape_even_without_inference(
    ontology_valid_store, ontology_shapes_store
):
    """`sh:targetClass` is subclass-aware even with `inference=None`.

    The empirically-confirmed fact `docs/ontology.md`/`KNOWN_ISSUES.md` rely on.
    """
    from rag.ontology.validation import run_shacl_validation

    report = run_shacl_validation(ontology_valid_store, ontology_shapes_store, inference=None)
    assert any("postgresql" in s for s in report.violation_summaries)


def test_invalid_fixture_trips_every_expected_shape(ontology_invalid_store, ontology_shapes_store):
    """Invalid fixture trips every expected shape."""
    from rag.ontology.validation import run_shacl_validation

    report = run_shacl_validation(ontology_invalid_store, ontology_shapes_store, inference=None)
    assert report.conforms is False
    assert report.violation_count == 7
    joined = "\n".join(report.violation_summaries)
    assert "orphan-service" in joined
    assert "exactly one tech:name" in joined
    assert "exactly one owning Team" in joined
    assert "exactly one customer" in joined
    assert "status must be exactly one of" in joined
    assert "tech:sourcePath" in joined
    assert "extractionMethod" in joined
    assert "team-workflow" in joined  # DependsOnEndpointsShape: Team is not Service/Application
