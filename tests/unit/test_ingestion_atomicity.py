"""Pipeline-level regression test for the CRITICAL ingestion-atomicity fix.

`test_pgvector_atomicity.py` proves `PgVectorStore.replace_document_chunks`
itself is transactionally atomic. This file proves the other half: that
`IngestionPipeline.ingest_file` actually calls that one atomic method for a
real write, instead of the old three separate,
independently-committing calls (`get_or_create_document_id`'s former
checksum write, `delete_chunks_by_document_id`, `add_chunks`).
"""

from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class _CallRecordingVectorStore:
    """VectorStore double that records which write methods are actually invoked.

    `add_chunks`/`delete_chunks_by_document_id` are still implemented (so
    `replace_document_chunks`'s own default-style delegation would still
    work if something called it that way), but each records its own call
    so a test can assert `ingest_file` never calls them *directly* --
    only through `replace_document_chunks`.
    """

    def __init__(self) -> None:
        """Start with no recorded documents, chunks, or calls."""
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}
        self._next_id = 0
        self.written_chunks: list[Chunk] = []
        self.calls: list[str] = []

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        """Assign a stable id per (source, dataset_id); never persists checksum itself."""
        self.calls.append("get_or_create_document_id")
        key = (source, dataset_id)
        if key not in self._documents:
            self._next_id += 1
            doc_id = f"doc-{self._next_id}"
            self._documents[key] = (doc_id, "")
            return doc_id, True
        doc_id, existing_checksum = self._documents[key]
        return doc_id, existing_checksum != checksum

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Record the call and remove any existing chunks for `document_id`."""
        self.calls.append("delete_chunks_by_document_id")
        self.written_chunks = [
            c for c in self.written_chunks if c.metadata.document_id != document_id
        ]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record the call and store `chunks`."""
        self.calls.append("add_chunks")
        self.written_chunks.extend(chunks)

    def replace_document_chunks(self, document_id: str, checksum: str, chunks: list[Chunk]) -> None:
        """Record the call, commit `checksum`, and atomically replace `document_id`'s chunks."""
        self.calls.append("replace_document_chunks")
        key = next(k for k, v in self._documents.items() if v[0] == document_id)
        self._documents[key] = (document_id, checksum)
        self.written_chunks = [
            c for c in self.written_chunks if c.metadata.document_id != document_id
        ]
        self.written_chunks.extend(chunks)


class _FakeEmbedder:
    """Minimal Embedder double returning a fixed placeholder vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, _CallRecordingVectorStore]:
    vectorstore = _CallRecordingVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=_FakeEmbedder())
    return pipeline, vectorstore


def test_new_document_ingestion_writes_only_through_replace_document_chunks(tmp_path: Path):
    """A first-time ingest calls replace_document_chunks, never add_chunks/delete directly."""
    path = tmp_path / "doc.md"
    path.write_text("Some fresh content to ingest.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    result = pipeline.ingest_file(path, "test-dataset")

    assert result["changed"] is True
    assert result["chunks_written"] > 0
    assert vectorstore.calls.count("replace_document_chunks") == 1
    assert "add_chunks" not in vectorstore.calls
    assert "delete_chunks_by_document_id" not in vectorstore.calls
    assert vectorstore.written_chunks  # the chunks really did land


def test_changed_document_reingestion_writes_only_through_replace_document_chunks(tmp_path: Path):
    """Re-ingesting a modified file also writes only through replace_document_chunks."""
    path = tmp_path / "doc.md"
    path.write_text("Original content.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_file(path, "test-dataset")
    vectorstore.calls.clear()

    path.write_text("Revised, longer content for the same file.", encoding="utf-8")
    result = pipeline.ingest_file(path, "test-dataset")

    assert result["changed"] is True
    assert vectorstore.calls == ["get_or_create_document_id", "replace_document_chunks"]
    assert any("Revised" in c.content for c in vectorstore.written_chunks)


def test_unchanged_document_never_calls_any_write_method(tmp_path: Path):
    """Re-ingesting an unmodified file calls get_or_create_document_id only, no write at all."""
    path = tmp_path / "doc.md"
    path.write_text("Stable content, never edited.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_file(path, "test-dataset")
    vectorstore.calls.clear()

    result = pipeline.ingest_file(path, "test-dataset")

    assert result["changed"] is False
    assert vectorstore.calls == ["get_or_create_document_id"]
