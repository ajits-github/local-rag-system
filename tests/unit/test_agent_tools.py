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
    """Records retrieve()/expand_with_relationships() calls and returns fixed results."""

    def __init__(self, retrieve_results=None, expand_results=None) -> None:
        """Store the fixed results this double's retrieve()/expand_with_relationships() returns."""
        self.retrieve_results = retrieve_results or []
        self.expand_results = expand_results or []
        self.retrieve_calls: list[dict] = []
        self.expand_calls: list[dict] = []

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        """Record the call and return the fixed retrieve results."""
        self.retrieve_calls.append(
            {"query": query, "filters": filters, "candidate_k": candidate_k, "auth": auth}
        )
        return self.retrieve_results

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
    """search_knowledge_base is a thin wrapper -- no retrieval logic of its own."""
    pipeline = FakePipeline(retrieve_results=[SearchResult(chunk=_chunk("c1"), score=0.9)])
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])
    args = SearchKnowledgeBaseArgs(query="hello", top_k=7)

    results = search_knowledge_base(args, pipeline, {"dataset_id": "ds1"}, auth)

    assert results == pipeline.retrieve_results
    assert pipeline.retrieve_calls == [
        {"query": "hello", "filters": {"dataset_id": "ds1"}, "candidate_k": 7, "auth": auth}
    ]


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
