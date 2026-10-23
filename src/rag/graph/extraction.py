"""Graph ingestion for one document: deterministic facts from metadata, plus optional LLM facts.

`extract_and_write_for_document` always writes the `Document`, `Category`,
and `SUPERSEDES` facts. These come from governance metadata and no model
is called. When `config.graph.extraction.llm_extraction_enabled` is set, it
also runs one LLM call per chunk.

LLM output is validated against a strict schema (`extra="forbid"`) whose
type fields use the `rag.graph.schema` allowlists, so the model cannot name
a type outside them. Relationships below `config.graph.extraction.min_confidence`
are dropped before any write.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.config import AppConfig
from rag.generation.base import LLM
from rag.graph.client import GraphUnavailableError, Neo4jClient
from rag.graph.schema import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    EntityType,
    GraphProvenance,
    GraphRelationship,
    RelationshipType,
)
from rag.graph.upsert import GraphWriter
from rag.prompts.loader import PromptTemplate
from rag.schemas import Chunk

logger = logging.getLogger(__name__)

# Bounds on the LLM's own JSON output, independent of
# config.graph.extraction.max_relationships_per_chunk (which further
# truncates after validation). This is a hard schema-level ceiling a
# malformed/runaway response can never exceed regardless of config.
_MAX_ENTITIES_PER_CALL = 20
_MAX_RELATIONSHIPS_PER_CALL = 20


class LLMExtractedEntity(BaseModel):
    """One entity in a `graph_extract_v1` LLM response."""

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    name: str = Field(max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)


class LLMExtractedRelationship(BaseModel):
    """One relationship in a `graph_extract_v1` LLM response."""

    model_config = ConfigDict(extra="forbid")

    subject_type: EntityType
    subject_name: str = Field(max_length=200)
    relationship_type: RelationshipType
    object_type: EntityType
    object_name: str = Field(max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMExtractionResult(BaseModel):
    """The full validated shape of one `graph_extract_v1` LLM response."""

    model_config = ConfigDict(extra="forbid")

    entities: list[LLMExtractedEntity] = Field(
        default_factory=list, max_length=_MAX_ENTITIES_PER_CALL
    )
    relationships: list[LLMExtractedRelationship] = Field(
        default_factory=list, max_length=_MAX_RELATIONSHIPS_PER_CALL
    )


class GraphExtractionStats(BaseModel):
    """Counts from one `extract_and_write_for_document` call.

    Attributes
    ----------
    llm_extraction_errors : int
        Chunks whose LLM call or parsing failed. Failures are counted here
        and never raised, so graph extraction cannot fail the ingestion.
    """

    deterministic_facts_written: int = 0
    llm_chunks_processed: int = 0
    llm_relationships_written: int = 0
    llm_relationships_discarded_low_confidence: int = 0
    llm_extraction_errors: int = 0


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_extraction(raw: str) -> LLMExtractionResult:
    """Parse and validate one `graph_extract_v1` response.

    Duplicates the JSON extraction in `rag.agent.decisions` on purpose,
    because `rag.graph` must not import `rag.agent`.

    Raises
    ------
    ValueError
        If `raw` contains no valid JSON, or it doesn't validate against
        `LLMExtractionResult`.
    """
    match = _JSON_OBJECT_RE.search(raw)
    candidate = match.group(0) if match else raw
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM extraction response is not valid JSON: {exc}") from exc
    try:
        return LLMExtractionResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM extraction response failed schema validation: {exc}") from exc


def extract_llm_facts(llm: LLM, template: PromptTemplate, chunk_text: str) -> LLMExtractionResult:
    """Run one LLM extraction call over `chunk_text`. Makes a single attempt, with no retry.

    A malformed response is discarded and an empty result is returned.

    Parameters
    ----------
    llm : LLM
        The LLM to call.
    template : PromptTemplate
        The `graph_extract_v1` (or later version) template.
    chunk_text : str
        The chunk's text content.

    Returns
    -------
    LLMExtractionResult
        The validated result, or an empty one if the call/parse failed.
    """
    system, user = template.render(
        entity_types=", ".join(sorted(ENTITY_TYPES)),
        relationship_types=", ".join(sorted(RELATIONSHIP_TYPES)),
        chunk_text=chunk_text,
    )
    try:
        raw = llm.generate(system, user)
        return _parse_llm_extraction(raw)
    except Exception:
        logger.warning("LLM graph extraction failed for one chunk", exc_info=True)
        return LLMExtractionResult()


def extract_and_write_for_document(
    writer: GraphWriter,
    client: Neo4jClient,
    config: AppConfig,
    llm: LLM | None,
    template: PromptTemplate | None,
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
    supersedes_document_id: str | None,
    chunks: list[Chunk],
) -> GraphExtractionStats:
    """Re-extract one document's graph facts, replacing any previous ones.

    Deletes this document's existing facts first, then writes the
    deterministic facts and, if enabled, the LLM facts. Failures are logged
    and counted, never raised. The Postgres ingestion must not fail because
    of graph extraction.

    Parameters
    ----------
    writer : GraphWriter
        The graph writer to issue upserts through.
    client : Neo4jClient
        The graph client, used only for the up-front connectivity check
        (see below). `writer` uses this same client internally for its
        own writes.
    config : AppConfig
        Application configuration; `config.graph.extraction` supplies
        every bound.
    llm : LLM | None
        The LLM for extraction calls; `None` disables LLM extraction
        regardless of config.
    template : PromptTemplate | None
        The `graph_extract_v1` template; `None` disables LLM extraction
        regardless of config.
    document_id, dataset_id, tenant_id, source, title, category,
    classification, status, document_version, trust_level, allowed_roles,
    checksum : see `GraphWriter.upsert_document_node`.
    supersedes_document_id : str | None
        The `document_id` this document supersedes, resolved by the caller.
    chunks : list[Chunk]
        The document's freshly written chunks. Provides `source_chunk_id`
        and the text for LLM extraction.

    Returns
    -------
    GraphExtractionStats
        Aggregate counts for this document's extraction run.
    """
    stats = GraphExtractionStats()
    if not client.health_check():
        logger.warning("Graph extraction skipped for %s: Neo4j unavailable", document_id)
        return stats

    try:
        writer.delete_facts_by_document(document_id)
        writer.upsert_document_node(
            document_id=document_id,
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            source=source,
            title=title,
            category=category,
            classification=classification,
            status=status,
            document_version=document_version,
            trust_level=trust_level,
            allowed_roles=allowed_roles,
            checksum=checksum,
        )
        stats.deterministic_facts_written += 1
        if category:
            writer.upsert_category_membership(
                document_id=document_id,
                dataset_id=dataset_id,
                tenant_id=tenant_id,
                category=category,
            )
            stats.deterministic_facts_written += 1
        if supersedes_document_id:
            writer.upsert_supersedes_edge(
                document_id=document_id,
                superseded_document_id=supersedes_document_id,
                dataset_id=dataset_id,
                tenant_id=tenant_id,
            )
            stats.deterministic_facts_written += 1
    except GraphUnavailableError:
        logger.warning("Deterministic graph extraction failed for %s", document_id, exc_info=True)
        return stats

    extraction_cfg = config.graph.extraction
    if not (extraction_cfg.llm_extraction_enabled and llm is not None and template is not None):
        return stats

    for chunk in chunks:
        try:
            result = extract_llm_facts(llm, template, chunk.content)
        except Exception:
            stats.llm_extraction_errors += 1
            continue
        stats.llm_chunks_processed += 1
        accepted = 0
        for rel in result.relationships:
            if accepted >= extraction_cfg.max_relationships_per_chunk:
                break
            if rel.confidence < extraction_cfg.min_confidence:
                stats.llm_relationships_discarded_low_confidence += 1
                continue
            provenance = GraphProvenance(
                source_document_id=document_id,
                source_chunk_id=chunk.metadata.chunk_id,
                dataset_id=dataset_id,
                tenant_id=tenant_id,
                allowed_roles=allowed_roles,
                trust_level=trust_level,
                extraction_method="llm",
                confidence=rel.confidence,
                document_checksum=checksum,
            )
            try:
                writer.upsert_relationship(
                    GraphRelationship(
                        subject_type=rel.subject_type,
                        subject_name=rel.subject_name,
                        relationship_type=rel.relationship_type,
                        object_type=rel.object_type,
                        object_name=rel.object_name,
                        provenance=provenance,
                    )
                )
                accepted += 1
                stats.llm_relationships_written += 1
            except Exception:
                logger.warning(
                    "Failed to write extracted relationship for %s", document_id, exc_info=True
                )
                stats.llm_extraction_errors += 1
    return stats
