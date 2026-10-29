"""MERGE-based graph writes, keyed by canonical id and provenance.

`GraphWriter` is the only module in `rag.graph` that issues write Cypher.
Every write re-checks entity and relationship types against the
`rag.graph.schema` allowlists, even though the Pydantic models already
enforce them.

Relationships are MERGEd on `(type, source_document_id, source_chunk_id)`.
Re-extracting the same fact from the same chunk is a no-op. The same fact
from two chunks creates two relationships, each removed when its own source
document is deleted. `delete_facts_by_document` removes a document's
relationships, then any node first seen in that document that has no
remaining relationships.

Entity ACL properties (`tenant_id`, `allowed_roles`, `trust_level`) are set
only `ON CREATE`, so they reflect the first document that mentioned the
entity; relationship ACLs are refreshed on every write. Graph evidence is
revalidated against the source document in Postgres at read time (see
`rag.graph.fusion`), so a stale entity ACL cannot widen what reaches a prompt.
"""

from __future__ import annotations

import logging

from rag.graph.canonicalization import canonicalize_display_name
from rag.graph.client import Neo4jClient
from rag.graph.schema import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    GraphEntity,
    GraphRelationship,
    canonical_entity_id,
)

logger = logging.getLogger(__name__)


class GraphSchemaViolationError(ValueError):
    """Raised when an entity/relationship type is not in the closed allowlist.

    Should be unreachable in practice (Pydantic's `Literal` typing already
    rejects an out-of-allowlist value at model-construction time); this is
    the defense-in-depth check for a caller that bypassed validation.
    """


class GraphWriter:
    """Issues idempotent `MERGE`-based writes to Neo4j, one entity/relationship at a time."""

    def __init__(self, client: Neo4jClient) -> None:
        """Store the injected `Neo4jClient`.

        Parameters
        ----------
        client : Neo4jClient
            The graph client to issue writes through.
        """
        self._client = client

    def upsert_document_node(
        self,
        *,
        document_id: str,
        dataset_id: str,
        tenant_id: str | None,
        source: str,
        title: str | None,
        category: str | None,
        classification: str | None,
        status: str | None,
        document_version: str | None,
        trust_level: str | None,
        allowed_roles: list[str] | None,
        checksum: str,
    ) -> None:
        """Upsert one `Document` node, keyed on the Postgres `document_id`.

        Unlike other entities, `Document` uses the existing `document_id` as
        its canonical id. The governance fields are copied from the document,
        so the same rules that gate its chunks also gate this node.

        Parameters
        ----------
        document_id : str
            The Postgres `documents.document_id`.
        dataset_id : str
            Dataset namespace.
        tenant_id : str | None
            Owning tenant, if any.
        source : str
            Relative document path.
        title : str | None
            Document title, if known.
        category : str | None
            Folder-derived category.
        classification : str | None
            Data-classification label.
        status : str | None
            Version lifecycle status.
        document_version : str | None
            Authored version label.
        trust_level : str | None
            Knowledge-source trust label.
        allowed_roles : list[str] | None
            Roles authorized to retrieve this document.
        checksum : str
            The document's content checksum at extraction time. Stored only.
        """
        cypher = """
        MERGE (d:Document {canonical_id: $document_id})
        SET d.document_id = $document_id,
            d.name = coalesce($title, $source),
            d.source = $source,
            d.title = $title,
            d.category = $category,
            d.dataset_id = $dataset_id,
            d.tenant_id = $tenant_id,
            d.classification = $classification,
            d.status = $status,
            d.document_version = $document_version,
            d.trust_level = $trust_level,
            d.allowed_roles = $allowed_roles,
            d.checksum = $checksum,
            d.first_seen_document_id = $document_id,
            d.updated_at = datetime()
        """
        self._client.run_write(
            cypher,
            {
                "document_id": document_id,
                "dataset_id": dataset_id,
                "tenant_id": tenant_id,
                "source": source,
                "title": title,
                "category": category,
                "classification": classification,
                "status": status,
                "document_version": document_version,
                "trust_level": trust_level,
                "allowed_roles": allowed_roles,
                "checksum": checksum,
            },
        )

    def upsert_category_membership(
        self, *, document_id: str, dataset_id: str, tenant_id: str | None, category: str
    ) -> None:
        """Upsert a `Category` node and a `Document-[:PART_OF]->Category` edge.

        `category` is the ingested file's folder path, taken from
        `ChunkMetadata.category`. No LLM is involved.

        Parameters
        ----------
        document_id : str
            The `Document` node's canonical id (== `documents.document_id`).
        dataset_id, tenant_id : str, str | None
            Scoping fields for the `Category` node's own `canonical_entity_id`.
        category : str
            The category name.
        """
        category_id = canonical_entity_id(
            "Category", category, dataset_id=dataset_id, tenant_id=tenant_id
        )
        cypher = """
        MATCH (d:Document {canonical_id: $document_id})
        MERGE (c:Category {canonical_id: $category_id})
        ON CREATE SET c.name = $category, c.dataset_id = $dataset_id, c.tenant_id = $tenant_id,
            c.first_seen_document_id = $document_id
        MERGE (d)-[r:PART_OF]->(c)
        SET r.dataset_id = $dataset_id, r.tenant_id = $tenant_id,
            r.source_document_id = $document_id, r.extraction_method = 'deterministic',
            r.confidence = 1.0
        """
        self._client.run_write(
            cypher,
            {
                "document_id": document_id,
                "category_id": category_id,
                "category": category,
                "dataset_id": dataset_id,
                "tenant_id": tenant_id,
            },
        )

    def upsert_supersedes_edge(
        self,
        *,
        document_id: str,
        superseded_document_id: str,
        dataset_id: str,
        tenant_id: str | None,
    ) -> None:
        """Upsert a `Document-[:SUPERSEDES]->Document` edge between two existing documents.

        The caller resolves `superseded_document_id` from `supersedes_source`,
        using the path-suffix rule in `retrieval/freshness.py`.

        Parameters
        ----------
        document_id : str
            The newer document's `document_id`.
        superseded_document_id : str
            The older, superseded document's `document_id`.
        dataset_id, tenant_id : str, str | None
            Provenance scoping fields.
        """
        cypher = """
        MATCH (newer:Document {canonical_id: $document_id})
        MATCH (older:Document {canonical_id: $superseded_document_id})
        MERGE (newer)-[r:SUPERSEDES]->(older)
        SET r.dataset_id = $dataset_id, r.tenant_id = $tenant_id,
            r.source_document_id = $document_id, r.extraction_method = 'deterministic',
            r.confidence = 1.0
        """
        self._client.run_write(
            cypher,
            {
                "document_id": document_id,
                "superseded_document_id": superseded_document_id,
                "dataset_id": dataset_id,
                "tenant_id": tenant_id,
            },
        )

    def upsert_entity(self, entity: GraphEntity) -> str:
        """Upsert an entity node without creating a relationship.

        Not called by the current extraction path. Kept for standalone writes.

        Parameters
        ----------
        entity : GraphEntity
            The validated entity to upsert.

        Returns
        -------
        str
            The entity's canonical id.

        Raises
        ------
        GraphSchemaViolationError
            If `entity.entity_type` is not in `ENTITY_TYPES`.
        """
        if entity.entity_type not in ENTITY_TYPES:
            raise GraphSchemaViolationError(f"Unknown entity type: {entity.entity_type}")
        name = canonicalize_display_name(entity.name)
        canonical_id = canonical_entity_id(
            entity.entity_type,
            name,
            dataset_id=entity.provenance.dataset_id,
            tenant_id=entity.provenance.tenant_id,
        )
        cypher = f"""
        MERGE (e:{entity.entity_type} {{canonical_id: $canonical_id}})
        ON CREATE SET e.name = $name, e.dataset_id = $dataset_id, e.tenant_id = $tenant_id,
            e.allowed_roles = $allowed_roles, e.trust_level = $trust_level,
            e.first_seen_document_id = $source_document_id
        SET e.aliases = coalesce(e.aliases, [])
            + [a IN $aliases WHERE NOT a IN coalesce(e.aliases, [])]
        """
        self._client.run_write(
            cypher,
            {
                "canonical_id": canonical_id,
                "name": name,
                "dataset_id": entity.provenance.dataset_id,
                "tenant_id": entity.provenance.tenant_id,
                "allowed_roles": entity.provenance.allowed_roles,
                "trust_level": entity.provenance.trust_level,
                "source_document_id": entity.provenance.source_document_id,
                "aliases": entity.aliases,
            },
        )
        return canonical_id

    def upsert_relationship(self, relationship: GraphRelationship) -> None:
        """Upsert both endpoint entities and the relationship between them.

        Each endpoint is MERGEd by `canonical_entity_id`. An existing node's
        properties are left unchanged, so a later document never overwrites
        an earlier one's provenance. The relationship is MERGEd on
        `(type, source_document_id, source_chunk_id)`.

        `source_chunk_id` is written as `""` when absent, because Neo4j
        rejects a `null` value in a MERGE key.

        Parameters
        ----------
        relationship : GraphRelationship
            The validated relationship to upsert.

        Raises
        ------
        GraphSchemaViolationError
            If any of `subject_type`/`object_type`/`relationship_type`
            is not in the closed allowlist.
        """
        if relationship.subject_type not in ENTITY_TYPES:
            raise GraphSchemaViolationError(f"Unknown entity type: {relationship.subject_type}")
        if relationship.object_type not in ENTITY_TYPES:
            raise GraphSchemaViolationError(f"Unknown entity type: {relationship.object_type}")
        if relationship.relationship_type not in RELATIONSHIP_TYPES:
            raise GraphSchemaViolationError(
                f"Unknown relationship type: {relationship.relationship_type}"
            )

        provenance = relationship.provenance
        subject_name = canonicalize_display_name(relationship.subject_name)
        object_name = canonicalize_display_name(relationship.object_name)
        subject_id = canonical_entity_id(
            relationship.subject_type,
            subject_name,
            dataset_id=provenance.dataset_id,
            tenant_id=provenance.tenant_id,
        )
        object_id = canonical_entity_id(
            relationship.object_type,
            object_name,
            dataset_id=provenance.dataset_id,
            tenant_id=provenance.tenant_id,
        )

        # The relationship type was checked against the allowlist above, so
        # it is safe to interpolate. Cypher cannot take it as a parameter.
        cypher = f"""
        MERGE (s:{relationship.subject_type} {{canonical_id: $subject_id}})
        ON CREATE SET s.name = $subject_name, s.dataset_id = $dataset_id, s.tenant_id = $tenant_id,
            s.allowed_roles = $allowed_roles, s.trust_level = $trust_level, s.aliases = [],
            s.first_seen_document_id = $source_document_id
        MERGE (o:{relationship.object_type} {{canonical_id: $object_id}})
        ON CREATE SET o.name = $object_name, o.dataset_id = $dataset_id, o.tenant_id = $tenant_id,
            o.allowed_roles = $allowed_roles, o.trust_level = $trust_level, o.aliases = [],
            o.first_seen_document_id = $source_document_id
        MERGE (s)-[r:{relationship.relationship_type} {{
            source_document_id: $source_document_id, source_chunk_id: $source_chunk_id
        }}]->(o)
        SET r.dataset_id = $dataset_id, r.tenant_id = $tenant_id,
            r.allowed_roles = $allowed_roles, r.trust_level = $trust_level,
            r.extraction_method = $extraction_method, r.confidence = $confidence,
            r.document_checksum = $document_checksum, r.extracted_at = $extracted_at
        """
        self._client.run_write(
            cypher,
            {
                "subject_id": subject_id,
                "subject_name": subject_name,
                "object_id": object_id,
                "object_name": object_name,
                "dataset_id": provenance.dataset_id,
                "tenant_id": provenance.tenant_id,
                "allowed_roles": provenance.allowed_roles,
                "trust_level": provenance.trust_level,
                "source_document_id": provenance.source_document_id,
                "source_chunk_id": provenance.source_chunk_id or "",
                "extraction_method": provenance.extraction_method,
                "confidence": provenance.confidence,
                "document_checksum": provenance.document_checksum,
                "extracted_at": provenance.extracted_at.isoformat(),
            },
        )

    def delete_facts_by_document(self, document_id: str) -> dict[str, int]:
        """Delete every graph fact sourced from `document_id`, then orphaned nodes.

        Called before a changed document is re-extracted, and when the ingestion
        pipeline removes a document. Both steps are idempotent:

        1. Delete relationships whose `source_document_id` matches.
        2. Delete nodes whose `first_seen_document_id` matches and that have no
           remaining relationships. A node still referenced by another
           document's facts is kept.

        Parameters
        ----------
        document_id : str
            The document whose graph facts should be removed.

        Returns
        -------
        dict[str, int]
            Combined write counters from both steps (see
            `Neo4jClient.run_write`).
        """
        delete_relationships_cypher = """
        MATCH ()-[r]->()
        WHERE r.source_document_id = $document_id
        DELETE r
        """
        delete_orphans_cypher = """
        MATCH (n)
        WHERE n.first_seen_document_id = $document_id AND NOT (n)--()
        DETACH DELETE n
        """
        rel_counters = self._client.run_write(
            delete_relationships_cypher, {"document_id": document_id}
        )
        node_counters = self._client.run_write(delete_orphans_cypher, {"document_id": document_id})
        return {
            "relationships_deleted": rel_counters["relationships_deleted"],
            "nodes_deleted": node_counters["nodes_deleted"],
        }

    def delete_facts_by_dataset(self, dataset_id: str) -> dict[str, int]:
        """Delete every node and relationship tagged with `dataset_id`.

        Called from `IngestionPipeline.clear_dataset`.

        Parameters
        ----------
        dataset_id : str
            The dataset namespace to clear.

        Returns
        -------
        dict[str, int]
            `{"nodes_deleted": int, "relationships_deleted": int}`.
        """
        counters = self._client.run_write(
            "MATCH (n) WHERE n.dataset_id = $dataset_id DETACH DELETE n", {"dataset_id": dataset_id}
        )
        return {
            "nodes_deleted": counters["nodes_deleted"],
            "relationships_deleted": counters["relationships_deleted"],
        }
