from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.tool_schemas import GetRelatedContextArgs, SearchKnowledgeBaseArgs
from rag.agent.tools import get_related_context, search_knowledge_base
from rag.retrieval.authorization import AuthorizationContext
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str, content: str = "content", document_id: str = "doc-1") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_id=chunk_id,
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


class FakePipeline:
    """Records retrieve()/resolve_auth()/expand_with_relationships() calls, returns fixed results.

    `resolve_auth` defaults to identity (returns `auth` unchanged) so tests
    that don't care about the authorization-enabled kill-switch/freshness
    resolution keep working unmodified; pass `resolve_auth_result` to
    prove a caller uses whatever `resolve_auth` returns rather than the
    raw `auth` it was given.
    """

    def __init__(self, retrieve_results=None, expand_results=None, resolve_auth_result=...) -> None:
        """Store the fixed results this double's retrieve()/expand_with_relationships() returns."""
        self.retrieve_results = retrieve_results or []
        self.expand_results = expand_results or []
        self._resolve_auth_result = resolve_auth_result
        self.retrieve_calls: list[dict] = []
        self.expand_calls: list[dict] = []
        self.resolve_auth_calls: list[dict] = []

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return the fixed retrieve results."""
        self.retrieve_calls.append(
            {"query": query, "filters": filters, "candidate_k": candidate_k, "auth": auth}
        )
        return self.retrieve_results

    def resolve_auth(self, auth, filters=None):
        """Record the call; return `resolve_auth_result` if set, else `auth` unchanged."""
        self.resolve_auth_calls.append({"auth": auth, "filters": filters})
        return auth if self._resolve_auth_result is ... else self._resolve_auth_result

    def expand_with_relationships(self, results, auth=None):
        """Record the call and append the fixed expand results onto the input list."""
        self.expand_calls.append({"results": results, "auth": auth})
        return list(results) + self.expand_results


class FakeVectorStore:
    """VectorStore double resolving get_chunks_by_ids() from a fixed id-to-Chunk mapping."""

    def __init__(self, chunks_by_id=None) -> None:
        """Store the fixed chunk_id -> Chunk mapping this double's get_chunks_by_ids() reads."""
        self._chunks_by_id = chunks_by_id or {}
        self.get_chunks_by_ids_calls: list[dict] = []

    def get_chunks_by_ids(self, chunk_ids, auth=None):
        """Record the call and return the chunks found in the fixed mapping."""
        self.get_chunks_by_ids_calls.append({"chunk_ids": chunk_ids, "auth": auth})
        return [self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id]


def test_search_knowledge_base_calls_pipeline_retrieve_with_top_k_as_candidate_k():
    """search_knowledge_base is a thin wrapper. no retrieval logic of its own."""
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk("c1"), score=0.9)])
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])
    args = SearchKnowledgeBaseArgs(query="hello", top_k=7)

    results = search_knowledge_base(args, pipeline, {"dataset_id": "ds1"}, auth)

    assert results == pipeline.retrieve_results
    assert pipeline.retrieve_calls == [
        {"query": "hello", "filters": {"dataset_id": "ds1"}, "candidate_k": 7, "auth": auth}
    ]


def test_search_knowledge_base_merges_content_type_into_filters():
    """A content_type arg is merged into the caller-supplied filters, not sent as a bare arg."""
    pipeline = FakePipeline(retrieve_results=[])
    args = SearchKnowledgeBaseArgs(query="find the table", content_type="table")

    search_knowledge_base(args, pipeline, {"dataset_id": "ds1"}, None)

    assert pipeline.retrieve_calls == [
        {
            "query": "find the table",
            "filters": {"dataset_id": "ds1", "content_type": "table"},
            "candidate_k": 5,
            "auth": None,
        }
    ]


def test_search_knowledge_base_omitted_content_type_leaves_filters_untouched():
    """No content_type arg. filters pass through exactly as the caller supplied them."""
    pipeline = FakePipeline(retrieve_results=[])
    args = SearchKnowledgeBaseArgs(query="hello")

    search_knowledge_base(args, pipeline, {"dataset_id": "ds1"}, None)

    assert pipeline.retrieve_calls[0]["filters"] == {"dataset_id": "ds1"}


def test_search_knowledge_base_content_type_works_with_no_caller_filters():
    """content_type still applies even when the caller passed no filters dict at all."""
    pipeline = FakePipeline(retrieve_results=[])
    args = SearchKnowledgeBaseArgs(query="find the diagram", content_type="image")

    search_knowledge_base(args, pipeline, None, None)

    assert pipeline.retrieve_calls[0]["filters"] == {"content_type": "image"}


def test_search_knowledge_base_content_type_rejects_arbitrary_values():
    """content_type is a closed Literal set. the LLM can't name an arbitrary filter value."""
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        SearchKnowledgeBaseArgs(query="hello", content_type="'; DROP TABLE chunks; --")


def test_get_related_context_expands_from_seed_chunk():
    """get_related_context fetches the seed chunk then reuses expand_with_relationships."""
    seed = _chunk("c1", content="seed")
    related = _chunk("c2", content="related")
    vectorstore = FakeVectorStore(chunks_by_id={"c1": seed})
    pipeline = FakePipeline(
        expand_results=[
            SearchResult(chunk=related, score=0.9, origin="expanded", expanded_from="c1")
        ]
    )
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    chunks = get_related_context(GetRelatedContextArgs(chunk_id="c1"), pipeline, vectorstore, auth)

    assert chunks == [related]
    assert vectorstore.get_chunks_by_ids_calls == [{"chunk_ids": ["c1"], "auth": auth}]
    assert pipeline.expand_calls[0]["auth"] == auth


def test_get_related_context_uses_resolved_auth_not_the_raw_auth_it_was_given():
    """The vectorstore and expansion calls receive resolve_auth's return value.

    A FakePipeline that returns a *different* object from resolve_auth
    than the auth it was given proves get_related_context forwards the
    resolved value, not a stale reference to the raw caller-supplied auth.
    """
    seed = _chunk("c1")
    vectorstore = FakeVectorStore(chunks_by_id={"c1": seed})
    resolved = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])
    pipeline = FakePipeline(resolve_auth_result=resolved)
    raw_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["operator"])

    get_related_context(GetRelatedContextArgs(chunk_id="c1"), pipeline, vectorstore, raw_auth)

    assert vectorstore.get_chunks_by_ids_calls == [{"chunk_ids": ["c1"], "auth": resolved}]
    assert pipeline.expand_calls[0]["auth"] == resolved
    assert pipeline.resolve_auth_calls == [{"auth": raw_auth, "filters": None}]


def test_get_related_context_threads_dataset_id_into_resolve_auth_filters():
    """A supplied dataset_id reaches resolve_auth as {"dataset_id": ...}.

    This lets get_related_context receive freshness-exclusion resolution
    when a dataset is known, matching get_document and get_latest_document.
    """
    seed = _chunk("c1")
    vectorstore = FakeVectorStore(chunks_by_id={"c1": seed})
    pipeline = FakePipeline()
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_related_context(GetRelatedContextArgs(chunk_id="c1"), pipeline, vectorstore, auth, "ds1")

    assert pipeline.resolve_auth_calls == [{"auth": auth, "filters": {"dataset_id": "ds1"}}]


def test_get_related_context_returns_empty_when_seed_not_found():
    """A chunk_id that doesn't resolve (missing, or excluded by auth) is empty, not an error."""
    vectorstore = FakeVectorStore(chunks_by_id={})
    pipeline = FakePipeline()
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    chunks = get_related_context(
        GetRelatedContextArgs(chunk_id="missing"), pipeline, vectorstore, auth
    )

    assert chunks == []
    assert pipeline.expand_calls == []


def test_get_related_context_only_returns_expanded_origin_chunks():
    """Only origin='expanded' results from expand_with_relationships are returned as related."""
    seed = _chunk("c1")
    vectorstore = FakeVectorStore(chunks_by_id={"c1": seed})
    expanded = _chunk("c2")
    pipeline = FakePipeline(
        expand_results=[
            SearchResult(chunk=expanded, score=0.9, origin="expanded", expanded_from="c1")
        ]
    )

    chunks = get_related_context(GetRelatedContextArgs(chunk_id="c1"), pipeline, vectorstore, None)

    assert chunks == [expanded]
    # The seed itself (origin="tool_fetched") must not be echoed back as "related".
    assert seed not in chunks
