from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class FakeVectorStore:
    """Minimal VectorStore double -- no DB, just records what was written."""

    def __init__(self) -> None:
        self.written_chunks: list[Chunk] = []
        self._next_id = 0

    def health_check(self) -> bool:
        return True

    def get_or_create_document_id(self, source: str, checksum: str) -> tuple[str, bool]:
        self._next_id += 1
        return f"doc-{self._next_id}", True

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        pass

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.written_chunks.extend(chunks)

    def search(self, *args, **kwargs):
        return []


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, FakeVectorStore]:
    vectorstore = FakeVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=FakeEmbedder())
    return pipeline, vectorstore


def test_ingest_path_derives_category_from_immediate_folder(tmp_path: Path):
    root = tmp_path / "kb"
    (root / "security").mkdir(parents=True)
    (root / "hr").mkdir(parents=True)
    (root / "security" / "a.md").write_text("Security policy content here.", encoding="utf-8")
    (root / "hr" / "b.md").write_text("HR policy content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root)

    categories = {c.metadata.source: c.metadata.category for c in vectorstore.written_chunks}
    assert categories[str(root / "security" / "a.md")] == "security"
    assert categories[str(root / "hr" / "b.md")] == "hr"


def test_ingest_path_derives_category_from_nested_folders(tmp_path: Path):
    root = tmp_path / "kb"
    nested = root / "security" / "subteam"
    nested.mkdir(parents=True)
    (nested / "c.md").write_text("Nested content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root)

    assert vectorstore.written_chunks[0].metadata.category == "security/subteam"


def test_ingest_single_file_has_no_category(tmp_path: Path):
    path = tmp_path / "standalone.md"
    path.write_text("Standalone content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(path)

    assert all(c.metadata.category is None for c in vectorstore.written_chunks)
