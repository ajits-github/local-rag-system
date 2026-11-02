"""Convert graph facts into `SearchResult`s so existing evidence code can handle them.

Each `GraphFact` becomes a `SearchResult` with `origin="graph"`. Its source
path and governance metadata (tenant, roles, classification, trust, status)
come from the documents that state the fact, read from Postgres, the
authoritative store, not from the ACL copy written into Neo4j. A fact is
dropped unless one derivation's documents all still exist in the dataset, are
visible to the caller, and still have the checksum the fact was projected from.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from rag.audit import log_audit_event
from rag.graph.retrieval import GraphFact, GraphFactSource
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.field_policy import detect_sensitive_field_ids
from rag.schemas import Chunk, ChunkMetadata, SearchResult
from rag.vectorstore.base import VectorStore

# Least to most restrictive. An unranked label sorts above all of these.
_CLASSIFICATION_RANK = {"public": 0, None: 1, "internal": 2, "confidential": 3, "restricted": 4}


def _combined_governance(sources: list[ChunkMetadata]) -> ChunkMetadata:
    """Merge a derivation's documents into metadata that satisfies the strictest of them.

    Citation fields come from the first document. A fact stated across several
    documents must not egress under the most permissive one's policy.
    """
    if len(sources) == 1:
        return sources[0]
    classification = max(
        (s.classification for s in sources),
        key=lambda label: _CLASSIFICATION_RANK.get(label, len(_CLASSIFICATION_RANK)),
    )
    restricted = [set(s.allowed_roles) for s in sources if s.allowed_roles]
    roles = sorted(set.intersection(*restricted)) if restricted else None
    return sources[0].model_copy(
        update={
            "classification": classification,
            # An empty intersection must stay a restriction: `[]` reads as unrestricted.
            "allowed_roles": roles or (sources[0].allowed_roles if restricted else None),
            "trust_level": next(
                (s.trust_level for s in sources if s.trust_level != "authoritative"),
                "authoritative",
            ),
            "status": next((s.status for s in sources if s.status != "active"), "active"),
        }
    )


def _derivations(fact: GraphFact) -> list[list[GraphFactSource]]:
    """Return a fact's alternative derivations; a node-only fact's is its source document."""
    if fact.support_sets:
        return fact.support_sets
    return [[GraphFactSource(document_id=fact.source_document_id)]]


def _resolve_sources(
    fact: GraphFact, governance: dict[str, ChunkMetadata], checksums: dict[str, str]
) -> list[ChunkMetadata] | None:
    """Return the metadata of the first derivation that is still valid, or `None`."""
    for sources in _derivations(fact):
        resolved = [governance.get(s.document_id or "") for s in sources]
        stale = any(
            s.document_checksum is not None
            and checksums.get(s.document_id or "") != s.document_checksum
            for s in sources
        )
        if sources and not stale and all(resolved):
            return [m for m in resolved if m is not None]
    return None


def graph_fact_to_search_result(
    fact: GraphFact, *, source_metadata: ChunkMetadata, score: float = 1.0
) -> SearchResult:
    """Wrap one `GraphFact` as a `SearchResult` with `origin="graph"`.

    Parameters
    ----------
    fact : GraphFact
        The normalized graph fact to wrap.
    source_metadata : ChunkMetadata
        Current Postgres metadata of the fact's source documents (see
        `VectorStore.get_document_governance`); supplies the citeable path
        and every governance field the egress and redaction checks read.
    score : float, optional
        A fixed ranking score for graph evidence (graph facts have no
        cosine-similarity concept); `GraphRAGFusionConfig` controls how
        many of these are kept, not how they're individually ranked
        against each other beyond stable extraction order.

    Returns
    -------
    SearchResult
        `origin="graph"`, `chunk.metadata.source_type="graph"` so
        `source_label()` and any downstream consumer can distinguish
        graph-derived evidence from a real retrieved chunk at a glance,
        even though the citation itself points at the same real document
        path a chunk from that document would. `sensitive_field_ids` is
        detected on the fact's own text, as ingestion does for a chunk.
        `chunk_id` is fact-specific (`graph:<source chunk or document>:<hash>`),
        never the source chunk's own id, so two facts from one chunk, or a
        fact and that retrieved chunk, never share an identity downstream.
    """
    document_id = source_metadata.document_id
    fact_key = "|".join([fact.fact_type, fact.text, *(e.canonical_id for e in fact.entities)])
    fact_hash = hashlib.sha256(fact_key.encode("utf-8")).hexdigest()[:16]
    chunk_id = f"graph:{fact.source_chunk_id or document_id}:{fact_hash}"
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_id=chunk_id,
        source=source_metadata.source,
        source_type="graph",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id=fact.dataset_id,
        content_type="graph_fact",
        sensitive_field_ids=detect_sensitive_field_ids(fact.text) or None,
        tenant_id=source_metadata.tenant_id,
        allowed_roles=source_metadata.allowed_roles,
        classification=source_metadata.classification,
        status=source_metadata.status,
        document_version=source_metadata.document_version,
        effective_from=source_metadata.effective_from,
        trust_level=source_metadata.trust_level,
        doc_source_type=source_metadata.doc_source_type,
    )
    return SearchResult(
        chunk=Chunk(id=chunk_id, content=fact.text, metadata=metadata),
        score=score,
        origin="graph",
    )


def graph_facts_to_search_results(
    facts: list[GraphFact],
    *,
    vectorstore: VectorStore,
    dataset_id: str,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Batch-convert `GraphFact`s, revalidating each against its source documents in Postgres.

    Parameters
    ----------
    facts : list[GraphFact]
        Facts to convert, in their existing order (already ranked/bounded
        by the caller, e.g. `GraphRAGPipeline`'s
        `config.graph.fusion.max_graph_evidence_items` cutoff).
    vectorstore : VectorStore
        Authoritative store for source-document existence and governance.
    dataset_id : str
        Dataset namespace.
    auth : AuthorizationContext | None
        The caller's context, already resolved via
        `RetrievalPipeline.resolve_auth` (kill-switch and freshness).

    Returns
    -------
    list[SearchResult]
        One `SearchResult` per fact with a still-valid derivation (see the
        module docstring), same order. Other facts are dropped (fail closed)
        and the drop count is audit-logged.
    """
    if not facts:
        return []
    sources = [s for fact in facts for derivation in _derivations(fact) for s in derivation]
    document_ids = [s.document_id for s in sources if s.document_id]
    governance = vectorstore.get_document_governance(document_ids, dataset_id, auth)
    pinned = [s.document_id for s in sources if s.document_id and s.document_checksum]
    checksums = vectorstore.get_document_checksums_by_id(pinned, dataset_id) if pinned else {}
    results = [
        graph_fact_to_search_result(fact, source_metadata=_combined_governance(resolved))
        for fact in facts
        if (resolved := _resolve_sources(fact, governance, checksums)) is not None
    ]
    dropped = len(facts) - len(results)
    if dropped:
        log_audit_event(
            "graph_facts_failed_revalidation", dataset_id=dataset_id, dropped_count=dropped
        )
    return results
