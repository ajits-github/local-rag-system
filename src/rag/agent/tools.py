"""Local agent tools: thin wrappers over existing retrieval, vector store, and graph services.

Tools contain no retrieval or authorization logic of their own. Each takes
`auth` as a separate parameter, taken from the run's
`AgentState.authorization_context`, never from parsed LLM output.

`get_document`, `get_latest_document`, and `get_related_context` query
`VectorStore` directly, so they resolve `auth` through
`RetrievalPipeline.resolve_auth` first; so do the graph tools, to revalidate
each fact against Postgres. `search_knowledge_base` does not, because
`retrieve()` resolves it internally.

The number of chunks loaded by the document tools is set by server config,
not by the model.
"""

from __future__ import annotations

import math

from rag.agent.tool_schemas import (
    FindDependencyPathArgs,
    GetDocumentArgs,
    GetEntityArgs,
    GetLatestDocumentArgs,
    GetNeighborsArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
    SearchKnowledgeGraphArgs,
)
from rag.embedders.base import Embedder
from rag.graph.fusion import graph_facts_to_search_results
from rag.graph.retrieval import GraphFact, GraphRetriever
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.freshness import resolve_current_document_source
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, DocumentVersionInfo, SearchResult
from rag.vectorstore.base import VectorStore


class ToolExecutionError(Exception):
    """Raised when a tool cannot complete its dispatch (e.g. missing dataset_id).

    Caught by the graph driver's `execute_tool` node and recorded as a
    failed `ToolCallRecord` rather than propagated; a tool failure must
    never crash the request.
    """


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors, or 0.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _select_relevant_chunks(
    chunks: list[Chunk], query: str, embedder: Embedder, max_chunks: int
) -> list[Chunk]:
    """Keep at most `max_chunks`, preferring those most relevant to `query`.

    If `chunks` already fits within `max_chunks`, returns it unchanged
    (still `chunk_index`-ordered, as fetched). Otherwise embeds `query`
    once and each candidate chunk's content (chunks fetched by id/section/
    source never carry a persisted `embedding`; see
    `PgVectorStore._METADATA_COLUMNS`), ranks by cosine similarity, and
    keeps the top `max_chunks`. Operates only on an already-fetched,
    already-authorized, already-hard-bounded batch. This is the same
    cosine-similarity operation `VectorStore.search` performs, applied
    client-side rather than in SQL, never a new retrieval mechanism.
    """
    if len(chunks) <= max_chunks:
        return chunks
    query_embedding = embedder.embed_query(query)
    chunk_embeddings = embedder.embed_documents([c.content for c in chunks])
    scored = list(zip(chunks, chunk_embeddings, strict=True))
    scored.sort(key=lambda pair: _cosine_similarity(query_embedding, pair[1]), reverse=True)
    return [chunk for chunk, _ in scored[:max_chunks]]


def search_knowledge_base(
    args: SearchKnowledgeBaseArgs,
    pipeline: RetrievalPipeline,
    filters: dict[str, object] | None,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Run authorized hybrid/dense retrieval via the existing `RetrievalPipeline`.

    Preserves every tenant/role/freshness/trust control `retrieve()`
    already applies; this function adds no retrieval logic of its own.
    When `args.content_type` is set, it is merged into the caller-supplied
    (server-controlled) `filters` dict: the LLM can only narrow to one of
    `ContentTypeFilter`'s allowed values via the whitelisted `content_type`
    field, never override or add any other key `filters` already carries
    (e.g. `dataset_id`).
    """
    effective_filters = dict(filters) if filters else {}
    if args.content_type is not None:
        effective_filters["content_type"] = args.content_type
    return pipeline.retrieve(
        args.query, filters=effective_filters or None, candidate_k=args.top_k, auth=auth
    )


def get_document(
    args: GetDocumentArgs,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    dataset_id: str | None,
    current_query: str,
    embedder: Embedder,
    auth: AuthorizationContext | None,
    max_chunks: int,
    max_chunks_hard_ceiling: int,
    versions: list[DocumentVersionInfo] | None = None,
) -> list[Chunk]:
    """Fetch a specific authorized document by its `source` path, bounded and relevance-selected.

    `auth` is resolved via `pipeline.resolve_auth` (kill-switch and
    freshness, scoped to `dataset_id`) before reaching `VectorStore`, the
    same as `search_knowledge_base`'s underlying `retrieve()` call.

    Parameters
    ----------
    versions : list[DocumentVersionInfo] | None, optional
        Pre-fetched `dataset_id` document versions, passed straight
        through to `pipeline.resolve_auth`. Lets a caller that resolves
        `auth` a second time for the same dispatch (e.g.
        `rag.agent.graph._execute_tool`'s post-dispatch
        `sanitize_evidence` call) share one `VectorStore.
        list_document_versions` fetch instead of triggering it twice.
        Fetched by `resolve_auth` itself when omitted, unchanged from
        prior behavior.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing. `get_chunks_by_source` requires it,
        and there is no safe default to guess.
    """
    if not dataset_id:
        raise ToolExecutionError("get_document requires a dataset_id filter")
    effective_auth = pipeline.resolve_auth(auth, {"dataset_id": dataset_id}, versions=versions)
    chunks = vectorstore.get_chunks_by_source(
        args.source, dataset_id, auth=effective_auth, limit=max_chunks_hard_ceiling
    )
    return _select_relevant_chunks(chunks, current_query, embedder, max_chunks)


def get_latest_document(
    args: GetLatestDocumentArgs,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    dataset_id: str | None,
    current_query: str,
    embedder: Embedder,
    auth: AuthorizationContext | None,
    max_chunks: int,
    max_chunks_hard_ceiling: int,
    versions: list[DocumentVersionInfo] | None = None,
) -> list[Chunk]:
    """Resolve `args.source` to its currently-effective version, then fetch it.

    If `args.source` names a superseded version, the current version's content
    is returned instead. `auth` is resolved through `pipeline.resolve_auth`,
    as in `get_document`.

    Parameters
    ----------
    versions : list[DocumentVersionInfo] | None, optional
        Pre-fetched document versions for `dataset_id`. Reused for source
        resolution and for `resolve_auth`. Fetched here when omitted.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing.
    """
    if not dataset_id:
        raise ToolExecutionError("get_latest_document requires a dataset_id filter")
    resolved_versions = (
        versions if versions is not None else vectorstore.list_document_versions(dataset_id)
    )
    resolved_source = resolve_current_document_source(args.source, resolved_versions)
    effective_auth = pipeline.resolve_auth(
        auth, {"dataset_id": dataset_id}, versions=resolved_versions
    )
    chunks = vectorstore.get_chunks_by_source(
        resolved_source, dataset_id, auth=effective_auth, limit=max_chunks_hard_ceiling
    )
    return _select_relevant_chunks(chunks, current_query, embedder, max_chunks)


def get_related_context(
    args: GetRelatedContextArgs,
    pipeline: RetrievalPipeline,
    vectorstore: VectorStore,
    auth: AuthorizationContext | None,
    dataset_id: str | None = None,
    versions: list[DocumentVersionInfo] | None = None,
) -> list[Chunk]:
    """Fetch parent/neighbor context for an already-retrieved chunk.

    Reuses `RetrievalPipeline.expand_with_relationships` directly rather
    than reimplementing parent/neighbor lookup; already bounded by
    `config.retrieval.relationship_expansion.max_related_elements`. A
    `chunk_id` that doesn't resolve (not found, or excluded by `auth`) is
    treated as a normal empty result, not a tool failure; nothing
    actually errored.

    `auth` is resolved once, via `pipeline.resolve_auth`, and the same
    resolved context is reused for both the seed chunk fetch and
    `expand_with_relationships`, mirroring how `retrieve()` resolves
    `auth` once and threads it through its own expansion call.

    Parameters
    ----------
    args : GetRelatedContextArgs
        Validated tool arguments (the seed chunk's `chunk_id`).
    pipeline : RetrievalPipeline
        Used for `resolve_auth` and `expand_with_relationships`.
    vectorstore : VectorStore
        Used for the seed chunk lookup.
    auth : AuthorizationContext | None
        Caller authorization context, resolved before use.
    dataset_id : str | None, optional
        The current query's dataset, when known. Enables freshness-
        exclusion resolution in `resolve_auth`, matching
        `get_document`/`get_latest_document`; when omitted, tenant/role
        enforcement still applies, just without freshness resolution.
    versions : list[DocumentVersionInfo] | None, optional
        Pre-fetched `dataset_id` document versions, passed straight
        through to `pipeline.resolve_auth`. Lets a caller that resolves
        `auth` a second time for the same dispatch (e.g.
        `rag.agent.graph._execute_tool`'s post-dispatch
        `sanitize_evidence` call) share one `VectorStore.
        list_document_versions` fetch instead of triggering it twice.
        Fetched by `resolve_auth` itself when omitted, unchanged from
        prior behavior.

    Returns
    -------
    list[Chunk]
        Related chunks (`origin == "expanded"` results only); empty if
        the seed chunk doesn't resolve or has no related content.
    """
    filters = {"dataset_id": dataset_id} if dataset_id else None
    effective_auth = pipeline.resolve_auth(auth, filters, versions=versions)
    seed_chunks = vectorstore.get_chunks_by_ids([args.chunk_id], auth=effective_auth)
    if not seed_chunks:
        return []
    seed_result = SearchResult(chunk=seed_chunks[0], score=1.0, origin="tool_fetched")
    expanded = pipeline.expand_with_relationships([seed_result], auth=effective_auth)
    return [r.chunk for r in expanded if r.origin == "expanded"]


def _revalidated_graph_results(
    facts: list[GraphFact],
    pipeline: RetrievalPipeline,
    dataset_id: str,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Convert facts to results, revalidated against Postgres under the resolved `auth`."""
    effective_auth = pipeline.resolve_auth(auth, {"dataset_id": dataset_id})
    return graph_facts_to_search_results(
        facts, vectorstore=pipeline.vectorstore, dataset_id=dataset_id, auth=effective_auth
    )


def search_knowledge_graph(
    args: SearchKnowledgeGraphArgs,
    graph_retriever: GraphRetriever,
    pipeline: RetrievalPipeline,
    dataset_id: str | None,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Run bounded graph retrieval for a natural-language question.

    Wraps `GraphRetriever.search`. The retriever applies the graph-side
    authorization checks; every fact is then revalidated against its source
    document in Postgres, so this tool returns only facts the caller may see.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing. Graph retrieval, like every other
        tool here, is always dataset-scoped.
    """
    if not dataset_id:
        raise ToolExecutionError("search_knowledge_graph requires a dataset_id filter")
    result = graph_retriever.search(args.query, dataset_id=dataset_id, auth=auth)
    return _revalidated_graph_results(result.facts, pipeline, dataset_id, auth)


def get_entity(
    args: GetEntityArgs,
    graph_retriever: GraphRetriever,
    pipeline: RetrievalPipeline,
    dataset_id: str | None,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Resolve a named entity and summarize its direct neighbors.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing.
    """
    if not dataset_id:
        raise ToolExecutionError("get_entity requires a dataset_id filter")
    result = graph_retriever.get_entity(
        args.name, dataset_id=dataset_id, auth=auth, entity_type=args.entity_type
    )
    return _revalidated_graph_results(result.facts, pipeline, dataset_id, auth)


def get_neighbors(
    args: GetNeighborsArgs,
    graph_retriever: GraphRetriever,
    pipeline: RetrievalPipeline,
    dataset_id: str | None,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Bounded-hop traversal from a named entity, optionally filtered by relationship type.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing.
    """
    if not dataset_id:
        raise ToolExecutionError("get_neighbors requires a dataset_id filter")
    result = graph_retriever.get_neighbors(
        args.entity_name,
        dataset_id=dataset_id,
        auth=auth,
        relationship_type=args.relationship_type,
        max_hops=args.max_hops,
    )
    return _revalidated_graph_results(result.facts, pipeline, dataset_id, auth)


def find_dependency_path(
    args: FindDependencyPathArgs,
    graph_retriever: GraphRetriever,
    pipeline: RetrievalPipeline,
    dataset_id: str | None,
    auth: AuthorizationContext | None,
) -> list[SearchResult]:
    """Bounded shortest path between two named entities.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing.
    """
    if not dataset_id:
        raise ToolExecutionError("find_dependency_path requires a dataset_id filter")
    result = graph_retriever.find_path(
        args.source_entity_name, args.target_entity_name, dataset_id=dataset_id, auth=auth
    )
    return _revalidated_graph_results(result.facts, pipeline, dataset_id, auth)
