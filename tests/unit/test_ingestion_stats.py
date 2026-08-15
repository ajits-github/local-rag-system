from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class StatefulFakeVectorStore:
    """Stateful VectorStore double tracking documents across repeated ingest_path calls.

    Unlike test_ingestion_category.py's FakeVectorStore (every file is
    always "new"), this one remembers checksums/chunk counts across calls,
    so it can exercise real new/changed/unchanged/deleted classification
    and the chunks_embedded/chunks_reused aggregate counts.
    """

    def __init__(self) -> None:
        """Start with no persisted documents."""
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}  # (source, ds) -> (id, sum)
        self._chunk_counts: dict[str, int] = {}
        self.written_chunks: list[Chunk] = []
        self._next_id = 0

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        """Assign a stable id per (source, dataset_id), reporting whether the checksum changed."""
        key = (source, dataset_id)
        if key not in self._documents:
            self._next_id += 1
            doc_id = f"doc-{self._next_id}"
            self._documents[key] = (doc_id, checksum)
            return doc_id, True
        doc_id, existing_checksum = self._documents[key]
        changed = existing_checksum != checksum
        if changed:
            self._documents[key] = (doc_id, checksum)
        return doc_id, changed

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Remove every written chunk for `document_id` and reset its chunk count."""
        self.written_chunks = [
            c for c in self.written_chunks if c.metadata.document_id != document_id
        ]
        self._chunk_counts.pop(document_id, None)

    def delete_document(self, document_id: str) -> None:
        """Remove a document's chunks (test double has no separate documents row)."""
        self.delete_chunks_by_document_id(document_id)

    def delete_dataset(self, dataset_id: str) -> None:
        """Remove every document tracked under `dataset_id`."""
        self._documents = {k: v for k, v in self._documents.items() if k[1] != dataset_id}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` and tally per-document chunk counts."""
        self.written_chunks.extend(chunks)
        for c in chunks:
            self._chunk_counts[c.metadata.document_id] = (
                self._chunk_counts.get(c.metadata.document_id, 0) + 1
            )

    def search(self, *args, **kwargs):
        """Return no results, always; unused by these tests."""
        return []

    def list_document_sources(self, dataset_id: str) -> list[str]:
        """List every source currently tracked under `dataset_id`."""
        return [source for source, ds in self._documents if ds == dataset_id]

    def delete_documents_by_source(self, dataset_id: str, sources: list[str]) -> int:
        """Delete documents by exact source match within `dataset_id`."""
        deleted = 0
        for source in sources:
            key = (source, dataset_id)
            if key in self._documents:
                doc_id, _ = self._documents.pop(key)
                self.delete_chunks_by_document_id(doc_id)
                deleted += 1
        return deleted

    def count_chunks_by_document(self, dataset_id: str) -> dict[str, int]:
        """Return `{document_id: chunk_count}` for every document in `dataset_id`."""
        ids = {
            doc_id for (_source, ds), (doc_id, _sum) in self._documents.items() if ds == dataset_id
        }
        return {doc_id: count for doc_id, count in self._chunk_counts.items() if doc_id in ids}


class FakeEmbedder:
    """Minimal Embedder double returning a fixed placeholder vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, StatefulFakeVectorStore]:
    """Build an IngestionPipeline wired to the stateful fake vectorstore/embedder."""
    vectorstore = StatefulFakeVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=FakeEmbedder())
    return pipeline, vectorstore


def test_first_ingestion_reports_every_file_as_new(tmp_path: Path):
    """A first-time directory ingest classifies every discovered file as 'new'."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content here.", encoding="utf-8")
    (root / "b.md").write_text("Beta content here.", encoding="utf-8")

    pipeline, _vs = _pipeline()
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.discovered == 2
    assert stats.new == 2
    assert stats.changed == 0
    assert stats.unchanged == 0
    assert stats.deleted == 0
    assert stats.chunks_embedded > 0
    assert stats.chunks_reused == 0
    assert {r["status"] for r in stats.results} == {"new"}


def test_reingesting_unchanged_files_reports_unchanged_and_reused_chunks(tmp_path: Path):
    """A second identical ingest reports every file 'unchanged' and sums their reused chunks."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "a.md").write_text("Alpha content here.", encoding="utf-8")

    pipeline, _vs = _pipeline()
    first = pipeline.ingest_path(root, "test-dataset")
    second = pipeline.ingest_path(root, "test-dataset")

    assert second.new == 0
    assert second.changed == 0
    assert second.unchanged == 1
    assert second.chunks_embedded == 0
    assert second.chunks_reused == first.chunks_embedded  # exactly what was written the first time


def test_modified_file_is_reported_as_changed_and_rewrites_chunks(tmp_path: Path):
    """Editing a file between ingests reports it as 'changed', re-embedding its chunks."""
    root = tmp_path / "kb"
    root.mkdir()
    path = root / "a.md"
    path.write_text("Alpha content here.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    path.write_text("Alpha content, now revised and longer.", encoding="utf-8")
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.new == 0
    assert stats.changed == 1
    assert stats.unchanged == 0
    assert stats.chunks_embedded > 0
    assert any("revised" in c.content for c in vs.written_chunks)


def test_deleted_file_is_detected_and_removed(tmp_path: Path):
    """A file removed from disk between ingests is detected and its document deleted."""
    root = tmp_path / "kb"
    root.mkdir()
    keep_path = root / "keep.md"
    remove_path = root / "remove.md"
    keep_path.write_text("Keep this content.", encoding="utf-8")
    remove_path.write_text("Remove this content.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    remove_path.unlink()
    stats = pipeline.ingest_path(root, "test-dataset")

    assert stats.discovered == 1  # only keep.md discovered this run
    assert stats.deleted == 1
    assert all("Remove this" not in c.content for c in vs.written_chunks)
    assert vs.list_document_sources("test-dataset") == [str(keep_path)]


def test_single_file_target_never_deletes_other_documents(tmp_path: Path):
    """Targeting a single file (not a directory) never triggers deleted-document detection."""
    root = tmp_path / "kb"
    root.mkdir()
    a_path = root / "a.md"
    b_path = root / "b.md"
    a_path.write_text("Alpha.", encoding="utf-8")
    b_path.write_text("Beta.", encoding="utf-8")

    pipeline, vs = _pipeline()
    pipeline.ingest_path(root, "test-dataset")
    stats = pipeline.ingest_path(a_path, "test-dataset")

    assert stats.deleted == 0
    assert len(vs.list_document_sources("test-dataset")) == 2  # b.md untouched
