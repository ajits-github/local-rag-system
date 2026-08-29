"""The four agent tools: thin wrappers over existing, tested application services.

No tool reimplements retrieval or security logic. Each calls
`RetrievalPipeline`/`VectorStore`/`rag.retrieval.freshness` directly. Every
function takes `auth: AuthorizationContext | None` as an explicit parameter
separate from its validated `args`, always supplied by the graph driver
from `AgentState.authorization_context`. Never constructed from parsed
LLM JSON (see `rag.agent.decisions`).

`get_document`/`get_latest_document`/`get_related_context` call
`VectorStore` directly rather than through `RetrievalPipeline.retrieve()`,
so each resolves `auth` via `RetrievalPipeline.resolve_auth` before passing
it to `VectorStore`, the same authorization-enabled kill-switch and
freshness resolution `retrieve()` already applies internally.
`search_knowledge_base` needs no such call: `pipeline.retrieve()` already
resolves `auth` itself.

`get_document`/`get_latest_document` bound how much of a document they load
into agent state: a server-controlled hard SQL `LIMIT`
(`max_chunks_hard_ceiling`) followed by a relevance-selection pass against
the current query (`max_chunks`). Neither number is LLM-writable, and
`rag.agent.tool_schemas.GetDocumentArgs`/`GetLatestDocumentArgs` expose no
chunk-count field at all.
"""

from __future__ import annotations

import math

from rag.agent.tool_schemas import (
    GetDocumentArgs,
    GetLatestDocumentArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
)
from rag.embedders.base import Embedder
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.freshness import resolve_current_document_source
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, SearchResult
from rag.vectorstore.base import VectorStore


class ToolExecutionError(Exception):
    """Raised when a tool cannot complete its dispatch (e.g. missing dataset_id).

    Caught by the graph driver's `execute_tool` node and recorded as a
    failed `ToolCallRecord` rather than propagated. A tool failure must
    never crash the request (see the "tool failure is handled safely"
    invariant in `docs/architecture.md`'s "Agentic RAG" section).
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
    When `args.content_type` is set, it's merged into the caller-supplied
    (server-controlled) `filters` dict -- the LLM can only ever narrow to
    one of `ContentTypeFilter`'s allowed values via the already-whitelisted
    `content_type` filter field (`VectorStore.ALLOWED_FILTER_FIELDS`),
    never override or add any other key `filters` already carries (e.g.
    `dataset_id`).
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
) -> list[Chunk]:
    """Fetch a specific authorized document by its `source` path, bounded and relevance-selected.

    `auth` is resolved via `pipeline.resolve_auth` (kill-switch and
    freshness, scoped to `dataset_id`) before reaching `VectorStore`, the
    same as `search_knowledge_base`'s underlying `retrieve()` call.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing. `get_chunks_by_source` requires it,
        and there is no safe default to guess.
    """
    if not dataset_id:
        raise ToolExecutionError("get_document requires a dataset_id filter")
    effective_auth = pipeline.resolve_auth(auth, {"dataset_id": dataset_id})
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
) -> list[Chunk]:
    """Resolve `args.source` to its currently-effective version, then fetch it.

    Unlike a plain `get_document(args.source)` call, this first resolves
    `args.source` through `resolve_current_document_source`. If
    `args.source` names a superseded family member, the *current* member's
    content is returned instead, which is the actual point of this tool
    (see `docs/architecture.md`'s "Agentic RAG" section on why a source
    path does not by itself identify the whole family).

    `auth` is resolved via `pipeline.resolve_auth` (kill-switch and
    freshness, scoped to `dataset_id`) before reaching `VectorStore`,
    identically to `get_document`. Reuses the `versions` list already
    fetched for source resolution rather than letting `resolve_auth`
    fetch it a second time.

    Raises
    ------
    ToolExecutionError
        If `dataset_id` is missing.
    """
    if not dataset_id:
        raise ToolExecutionError("get_latest_document requires a dataset_id filter")
    versions = vectorstore.list_document_versions(dataset_id)
    resolved_source = resolve_current_document_source(args.source, versions)
    effective_auth = pipeline.resolve_auth(auth, {"dataset_id": dataset_id}, versions=versions)
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
        The current query's dataset, when known (the graph driver's
        `state.filters.get("dataset_id")`). Passed through to
        `resolve_auth` so freshness-exclusion resolution applies here
        exactly as it does for `get_document`/`get_latest_document`;
        omitted only when no dataset scope is available, in which case
        `resolve_auth` still enforces tenant/role, just without
        freshness resolution (its own documented fallback).

    Returns
    -------
    list[Chunk]
        Related chunks (`origin == "expanded"` results only); empty if
        the seed chunk doesn't resolve or has no related content.
    """
    filters = {"dataset_id": dataset_id} if dataset_id else None
    effective_auth = pipeline.resolve_auth(auth, filters)
    seed_chunks = vectorstore.get_chunks_by_ids([args.chunk_id], auth=effective_auth)
    if not seed_chunks:
        return []
    seed_result = SearchResult(chunk=seed_chunks[0], score=1.0, origin="tool_fetched")
    expanded = pipeline.expand_with_relationships([seed_result], auth=effective_auth)
    return [r.chunk for r in expanded if r.origin == "expanded"]
