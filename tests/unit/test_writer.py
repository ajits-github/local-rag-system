from __future__ import annotations

from datetime import UTC, datetime

from rag.ingestion.writer import Writer
from rag.schemas import Chunk, ChunkSpan, RawDocument


class FakeVectorStore:
    """Minimal VectorStore double -- no DB, just records what was written."""

    def __init__(self) -> None:
        """Start with no recorded chunks."""
        self.written_chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` in `written_chunks` for assertions."""
        self.written_chunks.extend(chunks)


class FakeEmbedder:
    """Minimal Embedder double: returns a fixed placeholder vector per text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _raw_document() -> RawDocument:
    """Build a minimal RawDocument for writer tests."""
    now = datetime.now(UTC)
    return RawDocument(
        content="irrelevant",
        source="doc.md",
        source_type="markdown",
        created_at=now,
        last_modified=now,
    )


def test_span_with_no_content_type_defaults_to_prose():
    """A ChunkSpan that leaves content_type unset is persisted as 'prose', never None."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)

    chunks = writer.write(
        _raw_document(), "doc-1", [ChunkSpan(text="plain prose")], dataset_id="ds"
    )

    assert chunks[0].metadata.content_type == "prose"


def test_per_span_metadata_varies_chunk_to_chunk():
    """Each chunk's metadata reflects its own span, not the first/last span in the document."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [
        ChunkSpan(text="a table row", content_type="table", table_headers=["Name", "Value"]),
        ChunkSpan(text="def f(): pass", content_type="code", code_language="python"),
        ChunkSpan(text="plain prose paragraph"),
    ]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert [c.metadata.content_type for c in chunks] == ["table", "code", "prose"]
    assert chunks[0].metadata.table_headers == ["Name", "Value"]
    assert chunks[1].metadata.code_language == "python"
    assert chunks[2].metadata.table_headers is None
    assert chunks[2].metadata.code_language is None


def test_write_persists_chunks_to_vectorstore():
    """write() calls add_chunks with every constructed Chunk."""
    vectorstore = FakeVectorStore()
    writer = Writer(FakeEmbedder(), vectorstore)
    spans = [ChunkSpan(text="one"), ChunkSpan(text="two")]

    chunks = writer.write(_raw_document(), "doc-1", spans, dataset_id="ds")

    assert vectorstore.written_chunks == chunks
    assert [c.id for c in chunks] == ["doc-1_0", "doc-1_1"]
