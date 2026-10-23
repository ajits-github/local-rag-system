"""Convert graph facts into `SearchResult`s so existing evidence code can handle them.

Each `GraphFact` becomes a `SearchResult` with `origin="graph"`. Its `source`
is the real document path read from the fact's `Document` node, so citations
point at ingested content, not at a Neo4j id. Downstream context building,
citation, and redaction code therefore needs no special case.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.graph.client import Neo4jClient
from rag.graph.retrieval import GraphFact
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_UNKNOWN_SOURCE_PREFIX = "graph://unresolved/"


def resolve_document_sources(
    client: Neo4jClient, document_ids: list[str], dataset_id: str
) -> dict[str, str]:
    """Look up each `document_id`'s real `source` path from its `Document` node.

    One batched Cypher query rather than one per fact.

    Parameters
    ----------
    client : Neo4jClient
        Graph client to query through.
    document_ids : list[str]
        `source_document_id` values to resolve (deduplicated by the
        caller is not required; this function dedupes internally).
    dataset_id : str
        Dataset namespace, for defense-in-depth scoping (a fact's
        `source_document_id` should already only ever resolve within its
        own dataset, since `Document.canonical_id == document_id` is
        globally unique, but this still narrows the lookup).

    Returns
    -------
    dict[str, str]
        `{document_id: source_path}`, one entry per id that resolved to a
        real `Document` node. An id with no match is simply absent (never
        raises); callers fall back to a synthetic placeholder path.
    """
    unique_ids = sorted({doc_id for doc_id in document_ids if doc_id})
    if not unique_ids:
        return {}
    rows = client.run_read(
        "MATCH (d:Document) WHERE d.canonical_id IN $ids AND d.dataset_id = $dataset_id "
        "RETURN d.canonical_id AS document_id, d.source AS source",
        {"ids": unique_ids, "dataset_id": dataset_id},
    )
    return {row["document_id"]: row["source"] for row in rows if row.get("source")}


def graph_fact_to_search_result(
    fact: GraphFact, *, source_by_document_id: dict[str, str], score: float = 1.0
) -> SearchResult:
    """Wrap one `GraphFact` as a `SearchResult` with `origin="graph"`.

    Parameters
    ----------
    fact : GraphFact
        The normalized graph fact to wrap.
    source_by_document_id : dict[str, str]
        Output of `resolve_document_sources`, for resolving a real
        citeable path.
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
        path a chunk from that document would.
    """
    document_id = fact.source_document_id or "unknown"
    source = source_by_document_id.get(document_id, f"{_UNKNOWN_SOURCE_PREFIX}{document_id}")
    chunk_id = (
        fact.source_chunk_id
        or f"graph:{document_id}:{'|'.join(e.canonical_id for e in fact.entities)}"
    )
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        source_type="graph",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id=fact.dataset_id,
        content_type="graph_fact",
    )
    return SearchResult(
        chunk=Chunk(id=chunk_id, content=fact.text, metadata=metadata),
        score=score,
        origin="graph",
    )


def graph_facts_to_search_results(
    facts: list[GraphFact], client: Neo4jClient, dataset_id: str
) -> list[SearchResult]:
    """Batch-convert `GraphFact`s to `SearchResult`s, resolving real source paths once.

    Parameters
    ----------
    facts : list[GraphFact]
        Facts to convert, in their existing order (already ranked/bounded
        by the caller, e.g. `GraphRAGPipeline`'s
        `config.graph.fusion.max_graph_evidence_items` cutoff).
    client : Neo4jClient
        Graph client, for `resolve_document_sources`.
    dataset_id : str
        Dataset namespace.

    Returns
    -------
    list[SearchResult]
        One `SearchResult` per fact, same order, `origin="graph"`.
    """
    document_ids = [f.source_document_id for f in facts if f.source_document_id]
    source_by_document_id = resolve_document_sources(client, document_ids, dataset_id)
    return [
        graph_fact_to_search_result(fact, source_by_document_id=source_by_document_id)
        for fact in facts
    ]
