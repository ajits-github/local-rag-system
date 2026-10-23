from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rag.graph.schema import GraphEntity, GraphProvenance, GraphRelationship
from rag.graph.upsert import GraphSchemaViolationError, GraphWriter


class _FakeNeo4jClient:
    """Records every write call instead of talking to a real driver."""

    def __init__(self) -> None:
        self.write_calls: list[tuple[str, dict[str, Any]]] = []

    def run_write(
        self, cypher: str, parameters: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, int]:
        self.write_calls.append((cypher, parameters or {}))
        return {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_set": 0,
            "nodes_deleted": 0,
            "relationships_deleted": 0,
        }


def _provenance(**overrides: Any) -> GraphProvenance:
    defaults = dict(
        source_document_id="doc-1",
        source_chunk_id="doc-1_0",
        dataset_id="techfusion",
        tenant_id=None,
        allowed_roles=None,
        trust_level=None,
        extraction_method="deterministic",
        confidence=1.0,
        document_checksum="abc123",
        extracted_at=datetime(2026, 9, 30, tzinfo=UTC),
    )
    defaults.update(overrides)
    return GraphProvenance(**defaults)


def test_upsert_relationship_issues_one_write_with_both_endpoint_types_interpolated():
    """One write is issued, with both endpoints' labels and provenance correctly set."""
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    rel = GraphRelationship(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="PostgreSQL",
        provenance=_provenance(),
    )
    writer.upsert_relationship(rel)
    assert len(client.write_calls) == 1
    cypher, params = client.write_calls[0]
    assert "MERGE (s:Service" in cypher
    assert "MERGE (o:Database" in cypher
    assert "DEPENDS_ON" in cypher
    assert params["subject_name"] == "Document Service"
    assert params["object_name"] == "PostgreSQL"
    assert params["source_document_id"] == "doc-1"
    assert params["extraction_method"] == "deterministic"


def test_upsert_relationship_canonicalizes_known_aliases():
    """A known alias ('Postgres') resolves to its canonical display name before the write."""
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    rel = GraphRelationship(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="Postgres",  # alias, not the canonical "PostgreSQL"
        provenance=_provenance(),
    )
    writer.upsert_relationship(rel)
    _cypher, params = client.write_calls[0]
    assert params["object_name"] == "PostgreSQL"


def test_upsert_relationship_rejects_out_of_allowlist_type_defensively():
    """A relationship built via model_construct (bypassing Pydantic validation) is still rejected."""  # noqa: E501
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    rel = GraphRelationship.model_construct(
        subject_type="Service",
        subject_name="X",
        relationship_type="DROP_TABLE",  # not in RELATIONSHIP_TYPES
        object_type="Database",
        object_name="Y",
        provenance=_provenance(),
    )
    with pytest.raises(GraphSchemaViolationError):
        writer.upsert_relationship(rel)
    assert client.write_calls == []  # rejected before any write was attempted


def test_upsert_document_node_sets_canonical_id_to_the_document_id():
    """A `Document` node's `canonical_id` is set directly to the Postgres `document_id`."""
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    writer.upsert_document_node(
        document_id="doc-1",
        dataset_id="techfusion",
        tenant_id=None,
        source="architecture/system-overview.md",
        title="System Overview",
        category="architecture",
        classification=None,
        status="active",
        document_version=None,
        trust_level=None,
        allowed_roles=None,
        checksum="abc123",
    )
    cypher, params = client.write_calls[0]
    assert "MERGE (d:Document {canonical_id: $document_id})" in cypher
    assert params["document_id"] == "doc-1"
    assert params["source"] == "architecture/system-overview.md"


def test_delete_facts_by_document_issues_two_scoped_writes():
    """Deletion issues a relationship-delete write followed by an orphan-node-delete write."""
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    writer.delete_facts_by_document("doc-1")
    assert len(client.write_calls) == 2
    for _cypher, params in client.write_calls:
        assert params["document_id"] == "doc-1"
    assert "DELETE r" in client.write_calls[0][0]
    assert "DETACH DELETE n" in client.write_calls[1][0]


def test_upsert_entity_rejects_out_of_allowlist_type():
    """A standalone entity with an out-of-allowlist `entity_type` is rejected."""
    client = _FakeNeo4jClient()
    writer = GraphWriter(client)
    entity = GraphEntity.model_construct(
        entity_type="NotARealType", name="X", aliases=[], provenance=_provenance()
    )
    with pytest.raises(GraphSchemaViolationError):
        writer.upsert_entity(entity)
    assert client.write_calls == []
