"""Convert graph facts into `SearchResult`s so existing evidence code can handle them.

Each `GraphFact` becomes a `SearchResult` with `origin="graph"`. Its source
path and governance metadata (tenant, roles, classification, trust, status)
come from the fact's source document in Postgres, the authoritative store,
not from the ACL copy written into Neo4j. A fact whose source document is
deleted, outside the dataset, or no longer visible to the caller is dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.audit import log_audit_event
from rag.graph.retrieval import GraphFact
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.field_policy import detect_sensitive_field_ids
from rag.schemas import Chunk, ChunkMetadata, SearchResult
from rag.vectorstore.base import VectorStore


def graph_fact_to_search_result(
    fact: GraphFact, *, source_metadata: ChunkMetadata, score: float = 1.0
) -> SearchResult:
    """Wrap one `GraphFact` as a `SearchResult` with `origin="graph"`.

    Parameters
    ----------
    fact : GraphFact
        The normalized graph fact to wrap.
    source_metadata : ChunkMetadata
        Current Postgres metadata of the fact's source document (see
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
    """
    document_id = source_metadata.document_id
    chunk_id = (
        fact.source_chunk_id
        or f"graph:{document_id}:{'|'.join(e.canonical_id for e in fact.entities)}"
    )
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
    """Batch-convert `GraphFact`s, revalidating each against its source document in Postgres.

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
        One `SearchResult` per fact whose source document `auth` can still
        read, same order. Facts with no resolvable source are dropped
        (fail closed) and the drop count is audit-logged.
    """
    if not facts:
        return []
    document_ids = [f.source_document_id for f in facts if f.source_document_id]
    governance = vectorstore.get_document_governance(document_ids, dataset_id, auth)
    results = [
        graph_fact_to_search_result(fact, source_metadata=source_metadata)
        for fact in facts
        if (source_metadata := governance.get(fact.source_document_id or "")) is not None
    ]
    dropped = len(facts) - len(results)
    if dropped:
        log_audit_event(
            "graph_facts_failed_revalidation", dataset_id=dataset_id, dropped_count=dropped
        )
    return results
