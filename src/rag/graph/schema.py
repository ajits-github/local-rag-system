"""Closed entity/relationship type allowlists and the Pydantic models that carry graph facts.

Node labels and relationship types are fixed `Literal` sets. Both
`GraphWriter` and the LLM extraction schema use them, so the graph never
receives a type outside the allowlist. Every entity and relationship
carries `dataset_id` and `tenant_id` properties, which the Cypher
authorization predicate depends on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# The closed set of node labels this codebase writes to Neo4j. There is no
# "Person" label: the corpus names roles and teams, not individuals.
EntityType = Literal[
    "Document",
    "Category",
    "Tenant",
    "Service",
    "System",
    "Application",
    "Database",
    "API",
    "Team",
    "Policy",
    "Runbook",
    "Incident",
    "CustomerCase",
    "Customer",
    "Environment",
    "Technology",
    "Role",
]

ENTITY_TYPES: frozenset[str] = frozenset(EntityType.__args__)  # type: ignore[attr-defined]

# The closed set of relationship types this codebase writes. Each has one
# fixed meaning and is only created from an explicit source fact.
RelationshipType = Literal[
    "DEPENDS_ON",
    "OWNED_BY",
    "GOVERNED_BY",
    "APPLIES_TO",
    "RELATED_TO",
    "SUPERSEDES",
    "RUNS_ON",
    "CONNECTS_TO",
    "HAS_POLICY",
    "HAS_RUNBOOK",
    "AFFECTS",
    "REFERENCES",
    "PART_OF",
]

RELATIONSHIP_TYPES: frozenset[str] = frozenset(RelationshipType.__args__)  # type: ignore[attr-defined]

# Extraction methods recorded as provenance on every node/relationship,
# never inferred after the fact. See GraphProvenance.extraction_method.
ExtractionMethod = Literal["deterministic", "llm"]

_UNSCOPED_TENANT_BUCKET = "_shared"


def normalize_entity_name(name: str) -> str:
    """Lowercase and collapse whitespace in an entity name.

    Used only to derive `canonical_entity_id`; the node's displayed `name`
    is never normalized.

    Parameters
    ----------
    name : str
        Raw entity name, as extracted from a chunk or supplied by a query.

    Returns
    -------
    str
        Normalized name: lowercase, single-spaced, no leading/trailing
        whitespace.
    """
    return " ".join(name.split()).lower()


def canonical_entity_id(
    entity_type: str, name: str, *, dataset_id: str, tenant_id: str | None = None
) -> str:
    """Compute the deterministic canonical id used as the MERGE key for an entity.

    Scoped by entity type, dataset, tenant, and normalized name, so the
    same name in two datasets or two tenants yields two nodes.

    Parameters
    ----------
    entity_type : str
        One of `ENTITY_TYPES`. Not validated here; `GraphWriter` checks the
        allowlist before calling this.
    name : str
        The entity's display name, as extracted or queried.
    dataset_id : str
        The dataset namespace the entity was observed in.
    tenant_id : str | None, optional
        The owning tenant. `None` maps to a fixed `"_shared"` bucket that no
        real tenant id can collide with.

    Returns
    -------
    str
        `"{entity_type}:{dataset_id}:{tenant_bucket}:{normalized_name}"`.
    """
    tenant_bucket = tenant_id if tenant_id else _UNSCOPED_TENANT_BUCKET
    return f"{entity_type}:{dataset_id}:{tenant_bucket}:{normalize_entity_name(name)}"


class GraphProvenance(BaseModel):
    """Provenance stored as properties on every graph node and relationship.

    Attributes
    ----------
    source_document_id : str
        The Postgres `documents.document_id` this fact came from. Used to
        resolve a citeable source path.
    source_chunk_id : str | None
        The chunk a relationship was extracted from. `None` for facts that
        are not chunk-scoped, such as deterministic category membership.
    tenant_id : str | None
        Owning tenant, copied from the document.
    allowed_roles : list[str] | None
        Roles allowed to traverse this fact, copied from the document. `None`
        means unrestricted within the tenant and dataset scope.
    extraction_method : {"deterministic", "llm"}
        `"deterministic"` facts come from document metadata and always have
        `confidence=1.0`. `"llm"` facts come from schema-validated model output.
    confidence : float
        `1.0` for deterministic facts. For LLM facts, the model's self-reported
        confidence. Facts below `config.graph.extraction.min_confidence` are
        dropped before this object is built.
    document_checksum : str
        The source document's checksum at extraction time. Stored for
        provenance only; nothing currently compares it to detect staleness.
    extracted_at : datetime
        Extraction time in UTC, for observability.
    """

    source_document_id: str
    source_chunk_id: str | None = None
    dataset_id: str
    tenant_id: str | None = None
    allowed_roles: list[str] | None = None
    trust_level: str | None = None
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    document_checksum: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GraphEntity(BaseModel):
    """One node to upsert: a typed, named entity plus its provenance.

    Attributes
    ----------
    entity_type : EntityType
        One of `ENTITY_TYPES`.
    name : str
        Display name, as extracted or authored. Not normalized; `canonical_id`
        (computed via `canonical_entity_id`) is the actual MERGE key.
    aliases : list[str]
        Alternate names folded into the same node on upsert, so an
        abbreviation does not create a duplicate node.
    provenance : GraphProvenance
        Where this entity came from.
    """

    entity_type: EntityType
    name: str = Field(max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    provenance: GraphProvenance

    @property
    def canonical_id(self) -> str:
        """Return this entity's canonical id, computed from its own fields."""
        return canonical_entity_id(
            self.entity_type,
            self.name,
            dataset_id=self.provenance.dataset_id,
            tenant_id=self.provenance.tenant_id,
        )


class GraphRelationship(BaseModel):
    """One relationship to upsert, between two entities identified by (type, name).

    Subject/object are identified by type+name (not a pre-resolved
    canonical id) so extraction code never has to compute
    `canonical_entity_id` itself. `GraphWriter.upsert_relationship`
    does, from the same `dataset_id`/`tenant_id` on `provenance`, so the
    scoping rule lives in exactly one place (`rag.graph.schema`).

    Attributes
    ----------
    subject_type, subject_name : EntityType, str
        The relationship's source entity.
    relationship_type : RelationshipType
        One of `RELATIONSHIP_TYPES`.
    object_type, object_name : EntityType, str
        The relationship's target entity.
    provenance : GraphProvenance
        Where this relationship came from.
    """

    subject_type: EntityType
    subject_name: str = Field(max_length=200)
    relationship_type: RelationshipType
    object_type: EntityType
    object_name: str = Field(max_length=200)
    provenance: GraphProvenance
