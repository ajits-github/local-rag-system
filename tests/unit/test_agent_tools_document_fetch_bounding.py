from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.agent.tool_schemas import GetDocumentArgs
from rag.agent.tools import ToolExecutionError, get_document
from rag.schemas import Chunk, ChunkMetadata


def _chunk(chunk_id: str, content: str) -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source="policy.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=int(chunk_id.split("-")[-1]),
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


class FakeVectorStore:
    """Fake get_chunks_by_source() that ignores real `limit` semantics.

    The fake just returns however many the test wants, and the test
    asserts the tool *requested* the hard ceiling as `limit`.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Store the fixed chunk list this double's get_chunks_by_source() slices from."""
        self._chunks = chunks
        self.calls: list[dict] = []

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        """Record the call and return the fixed chunks, sliced to `limit`."""
        self.calls.append(
            {"source": source, "dataset_id": dataset_id, "auth": auth, "limit": limit}
        )
        return self._chunks[:limit] if limit is not None else list(self._chunks)


class FakeEmbedder:
    """Deterministic embedder: each chunk's "embedding" is a one-hot vector by its keyword."""

    def __init__(self) -> None:
        """Start with no recorded embed_query()/embed_documents() calls."""
        self.embed_query_calls: list[str] = []
        self.embed_documents_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        """Record the call and return a one-hot vector keyed on whether "alpha" is present."""
        self.embed_query_calls.append(text)
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Record the call and return one one-hot vector per text, keyed on "alpha"."""
        self.embed_documents_calls.append(texts)
        return [[1.0, 0.0] if "alpha" in t else [0.0, 1.0] for t in texts]


def test_get_document_returns_all_chunks_unchanged_when_within_limit():
    """When the fetch already fits max_chunks, no relevance-selection/embedding call happens."""
    chunks = [_chunk("doc-1_0", "alpha content"), _chunk("doc-1_1", "beta content")]
    vectorstore = FakeVectorStore(chunks)
    embedder = FakeEmbedder()

    result = get_document(
        GetDocumentArgs(source="policy.md"),
        vectorstore,
        "ds1",
        "current query",
        embedder,
        None,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert result == chunks
    assert embedder.embed_query_calls == []
    assert embedder.embed_documents_calls == []


def test_get_document_passes_hard_ceiling_as_the_sql_limit():
    """The SQL-level fetch always uses the hard ceiling, not the smaller final chunk count."""
    vectorstore = FakeVectorStore([_chunk("doc-1_0", "content")])
    embedder = FakeEmbedder()

    get_document(
        GetDocumentArgs(source="policy.md"),
        vectorstore,
        "ds1",
        "query",
        embedder,
        None,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert vectorstore.calls[0]["limit"] == 50


def test_get_document_relevance_selects_when_over_max_chunks():
    """A fetch exceeding max_chunks is cosine-ranked against the current query and truncated."""
    chunks = [
        _chunk("doc-1_0", "alpha alpha alpha"),
        _chunk("doc-1_1", "beta beta beta"),
        _chunk("doc-1_2", "beta beta content"),
    ]
    vectorstore = FakeVectorStore(chunks)
    embedder = FakeEmbedder()

    result = get_document(
        GetDocumentArgs(source="policy.md"),
        vectorstore,
        "ds1",
        "alpha question",
        embedder,
        None,
        max_chunks=1,
        max_chunks_hard_ceiling=50,
    )

    assert len(result) == 1
    assert result[0].metadata.chunk_id == "doc-1_0"  # the alpha-matching chunk wins
    assert embedder.embed_query_calls == ["alpha question"]


def test_get_document_requires_dataset_id():
    """Missing dataset_id is a safe ToolExecutionError, not a crash -- caught by the graph."""
    vectorstore = FakeVectorStore([])
    embedder = FakeEmbedder()

    with pytest.raises(ToolExecutionError):
        get_document(
            GetDocumentArgs(source="policy.md"),
            vectorstore,
            None,
            "query",
            embedder,
            None,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
