from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class FakeVectorStore:
    """Minimal VectorStore double. no DB, just records what was written."""

    def __init__(self) -> None:
        """Start with no recorded chunks and a fresh document-id counter."""
        self.written_chunks: list[Chunk] = []
        self._next_id = 0

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_or_create_document_id(
        self, source: str, checksum: str, dataset_id: str
    ) -> tuple[str, bool]:
        """Assign a new incrementing document id, always reporting `changed`."""
        self._next_id += 1
        return f"doc-{self._next_id}", True

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """No-op: nothing is persisted to delete from."""

    def delete_document(self, document_id: str) -> None:
        """No-op: nothing is persisted to delete from."""

    def delete_dataset(self, dataset_id: str) -> None:
        """No-op: nothing is persisted to delete from."""

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Record `chunks` in `written_chunks` for assertions."""
        self.written_chunks.extend(chunks)

    def search(self, *args, **kwargs):
        """Return no results, always; unused by these tests."""
        return []

    def list_document_sources(self, dataset_id: str) -> list[str]:
        """Report no pre-existing documents. every ingest_path call here starts fresh."""
        return []

    def delete_documents_by_source(self, dataset_id: str, sources: list[str]) -> int:
        """No-op: these tests never exercise deleted-document detection."""
        return 0

    def count_chunks_by_document(self, dataset_id: str) -> dict[str, int]:
        """No-op: these tests never exercise the chunks_reused stat."""
        return {}


class FakeEmbedder:
    """Minimal Embedder double: returns a fixed placeholder vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


def _pipeline() -> tuple[IngestionPipeline, FakeVectorStore]:
    """Build an IngestionPipeline wired to fake vectorstore/embedder doubles."""
    vectorstore = FakeVectorStore()
    pipeline = IngestionPipeline(load_config(), vectorstore=vectorstore, embedder=FakeEmbedder())
    return pipeline, vectorstore


def test_ingest_path_derives_category_from_immediate_folder(tmp_path: Path):
    """A file's immediate parent folder becomes its chunk category."""
    root = tmp_path / "kb"
    (root / "security").mkdir(parents=True)
    (root / "hr").mkdir(parents=True)
    (root / "security" / "a.md").write_text("Security policy content here.", encoding="utf-8")
    (root / "hr" / "b.md").write_text("HR policy content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    categories = {c.metadata.source: c.metadata.category for c in vectorstore.written_chunks}
    assert categories[str(root / "security" / "a.md")] == "security"
    assert categories[str(root / "hr" / "b.md")] == "hr"


def test_ingest_path_derives_category_from_nested_folders(tmp_path: Path):
    """A nested folder path becomes a "/"-joined category, e.g. "security/subteam"."""
    root = tmp_path / "kb"
    nested = root / "security" / "subteam"
    nested.mkdir(parents=True)
    (nested / "c.md").write_text("Nested content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    assert vectorstore.written_chunks[0].metadata.category == "security/subteam"


def test_ingest_single_file_has_no_category(tmp_path: Path):
    """Ingesting a single file (no directory walk) leaves category unset."""
    path = tmp_path / "standalone.md"
    path.write_text("Standalone content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(path, "test-dataset")

    assert all(c.metadata.category is None for c in vectorstore.written_chunks)


def test_ingest_path_tags_every_chunk_with_dataset_id(tmp_path: Path):
    """Every chunk written during ingestion carries the given dataset_id."""
    root = tmp_path / "kb"
    (root / "security").mkdir(parents=True)
    (root / "security" / "a.md").write_text("Security policy content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "techfusion")

    assert vectorstore.written_chunks  # sanity: something was written
    assert all(c.metadata.dataset_id == "techfusion" for c in vectorstore.written_chunks)


def test_ingest_path_different_dataset_ids_are_independent(tmp_path: Path):
    """Ingesting the same path under two dataset_ids keeps both tag sets distinct."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "a.md").write_text("Shared-looking content here.", encoding="utf-8")

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "dataset-a")
    pipeline.ingest_path(root, "dataset-b")

    dataset_ids = {c.metadata.dataset_id for c in vectorstore.written_chunks}
    assert dataset_ids == {"dataset-a", "dataset-b"}


def test_ingest_table_chunk_tagged_content_type_table(tmp_path: Path):
    """A Markdown table in an ingested file produces a chunk with content_type='table'."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "doc.md").write_text(
        "# Report\n\n| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |\n",
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    table_chunks = [c for c in vectorstore.written_chunks if c.metadata.content_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].metadata.table_headers == ["Name", "Value"]
    assert table_chunks[0].metadata.section_path == "Report"


def test_ingest_code_chunk_tagged_content_type_code_with_language(tmp_path: Path):
    """A fenced Python block in an ingested file produces a chunk tagged content_type='code'."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "doc.md").write_text(
        "# Runbook\n\nRun this:\n\n```python\ndef retry():\n    return True\n```\n",
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    code_chunks = [c for c in vectorstore.written_chunks if c.metadata.content_type == "code"]
    assert len(code_chunks) == 1
    assert code_chunks[0].metadata.code_language == "python"
    assert code_chunks[0].metadata.section_path == "Runbook"


def test_ingest_configuration_chunk_tagged_content_type_configuration(tmp_path: Path):
    """A fenced JSON block in an ingested file produces a content_type='configuration' chunk."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "doc.md").write_text(
        '# Config\n\n```json\n{"retries": 3}\n```\n',
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    config_chunks = [
        c for c in vectorstore.written_chunks if c.metadata.content_type == "configuration"
    ]
    assert len(config_chunks) == 1
    assert config_chunks[0].metadata.code_language == "json"


def test_ingest_chart_caption_chunk_tagged_content_type_chart(tmp_path: Path):
    """A ```text fence + emphasis caption in an ingested file produces content_type='chart'."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "doc.md").write_text(
        "# Capacity\n\n```text\nQ1 |####\nQ2 |######\n```\n\n*Chart caption: usage grew.*\n",
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    chart_chunks = [c for c in vectorstore.written_chunks if c.metadata.content_type == "chart"]
    assert len(chart_chunks) == 1
    assert "Chart caption: usage grew." in chart_chunks[0].content


def test_ingest_mixed_prose_and_table_preserves_section_context(tmp_path: Path):
    """Prose before/after a table, and the table itself, all keep the same section_path."""
    root = tmp_path / "kb"
    root.mkdir(parents=True)
    (root / "doc.md").write_text(
        "# Metrics\n\nSome intro prose about metrics.\n\n"
        "| Name | Value |\n|---|---|\n| a | 1 |\n\n"
        "Some closing prose about metrics.\n",
        encoding="utf-8",
    )

    pipeline, vectorstore = _pipeline()
    pipeline.ingest_path(root, "test-dataset")

    chunks = vectorstore.written_chunks
    assert chunks  # sanity
    assert all(c.metadata.section_path == "Metrics" for c in chunks)
    content_types = {c.metadata.content_type for c in chunks}
    assert "table" in content_types
    assert "prose" in content_types
