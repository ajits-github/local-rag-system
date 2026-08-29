from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.agent.tool_schemas import GetLatestDocumentArgs
from rag.agent.tools import ToolExecutionError, get_latest_document
from rag.schemas import Chunk, ChunkMetadata, DocumentVersionInfo


def _chunk(chunk_id: str, source: str, content: str = "content") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-2",
        chunk_id=chunk_id,
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


class FakeVectorStore:
    """VectorStore double resolving versions and chunks-by-source from fixed fixture data."""

    def __init__(self, versions: list[DocumentVersionInfo], chunks_by_source: dict) -> None:
        """Store the fixed version list and source-to-chunks mapping this double reads from."""
        self._versions = versions
        self._chunks_by_source = chunks_by_source
        self.get_chunks_by_source_calls: list[dict] = []

    def list_document_versions(self, dataset_id):
        """Return the fixed version list, ignoring dataset_id."""
        return self._versions

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        """Record the call and return the fixed chunks for `source`."""
        self.get_chunks_by_source_calls.append(
            {"source": source, "dataset_id": dataset_id, "auth": auth, "limit": limit}
        )
        return self._chunks_by_source.get(source, [])


class FakePipeline:
    """resolve_auth() double returning `auth` unchanged (identity); records every call.

    Authorization parity tests exercise the real kill-switch/freshness
    logic via a real RetrievalPipeline; this identity fake keeps these
    version-resolution tests focused on their own concern.
    """

    def __init__(self) -> None:
        """Start with no recorded resolve_auth() calls."""
        self.resolve_auth_calls: list[dict] = []

    def resolve_auth(self, auth, filters=None, *, versions=None):
        """Record the call and return `auth` unchanged."""
        self.resolve_auth_calls.append({"auth": auth, "filters": filters, "versions": versions})
        return auth


class FakeEmbedder:
    """Minimal Embedder double returning fixed placeholder vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text; unused here."""
        return [[1.0] for _ in texts]


def test_get_latest_document_redirects_a_superseded_source_to_the_current_one():
    """Requesting an old version's path still returns the currently-active member's content.

    This is the corrected design from the plan review (adjustment 7):
    get_chunks_by_source alone on the *requested* path would silently
    return stale content. get_latest_document must resolve the family
    first.
    """
    versions = [
        DocumentVersionInfo(document_id="doc-1", source="policy-v1.md", status="superseded"),
        DocumentVersionInfo(
            document_id="doc-2",
            source="policy-v2.md",
            status="active",
            supersedes_source="policy-v1.md",
        ),
    ]
    current_chunks = [_chunk("doc-2_0", "policy-v2.md", "current content")]
    vectorstore = FakeVectorStore(
        versions, chunks_by_source={"policy-v2.md": current_chunks, "policy-v1.md": []}
    )
    pipeline = FakePipeline()
    embedder = FakeEmbedder()

    result = get_latest_document(
        GetLatestDocumentArgs(source="policy-v1.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        embedder,
        None,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert result == current_chunks
    # The fetch happened against the *resolved* source, not the requested one.
    assert vectorstore.get_chunks_by_source_calls[0]["source"] == "policy-v2.md"


def test_get_latest_document_uses_hard_ceiling_as_limit():
    """The SQL-level fetch always uses max_chunks_hard_ceiling as its `limit`."""
    versions = [DocumentVersionInfo(document_id="doc-1", source="standalone.md", status="active")]
    vectorstore = FakeVectorStore(versions, chunks_by_source={"standalone.md": []})
    pipeline = FakePipeline()
    embedder = FakeEmbedder()

    get_latest_document(
        GetLatestDocumentArgs(source="standalone.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        embedder,
        None,
        max_chunks=10,
        max_chunks_hard_ceiling=42,
    )

    assert vectorstore.get_chunks_by_source_calls[0]["limit"] == 42


def test_get_latest_document_requires_dataset_id():
    """Missing dataset_id is a safe ToolExecutionError, not a crash."""
    vectorstore = FakeVectorStore([], chunks_by_source={})
    pipeline = FakePipeline()
    embedder = FakeEmbedder()

    with pytest.raises(ToolExecutionError):
        get_latest_document(
            GetLatestDocumentArgs(source="a.md"),
            pipeline,
            vectorstore,
            None,
            "query",
            embedder,
            None,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
