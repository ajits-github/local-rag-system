"""Durable graph-projection retry (BUG-015).

A graph write that failed after the vector commit used to be permanent: the
unchanged-checksum shortcut returned before graph extraction on every later
re-ingest. `documents.graph_checksum` now records the last successful
projection, and an unchanged re-ingest repairs a missing or stale one.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.ingestion import pipeline as pipeline_module
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class _VectorStore:
    """Tracks document checksums, graph checksums, and stored chunks in memory."""

    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}
        self.graph_checksums: dict[str, str | None] = {}
        self.chunks: list[Chunk] = []

    def get_or_create_document_id(self, source, checksum, dataset_id):
        key = (source, dataset_id)
        if key not in self._documents:
            self._documents[key] = (f"doc-{len(self._documents) + 1}", "")
            return self._documents[key][0], True
        doc_id, existing = self._documents[key]
        return doc_id, existing != checksum

    def replace_document_chunks(self, document_id, checksum, chunks):
        key = next(k for k, v in self._documents.items() if v[0] == document_id)
        self._documents[key] = (document_id, checksum)
        self.chunks = list(chunks)

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        return [c for c in self.chunks if c.metadata.source == source]

    def get_graph_checksum(self, document_id):
        return self.graph_checksums.get(document_id)

    def set_graph_checksum(self, document_id, checksum):
        self.graph_checksums[document_id] = checksum

    def clear_graph_checksums(self, dataset_id):
        self.graph_checksums = {k: None for k in self.graph_checksums}

    def list_document_versions(self, dataset_id):
        return []


class _Embedder:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _GraphClient:
    """Stand-in `Neo4jClient`; extraction itself is replaced by `_Extraction`."""

    def run_write(self, *args, **kwargs):
        return {}


class _Extraction:
    """Replaces `extract_and_write_for_document`; fails while `neo4j_down` is set."""

    def __init__(self) -> None:
        self.neo4j_down = False
        self.calls: list[int] = []

    def __call__(self, *args, chunks, **kwargs):
        self.calls.append(len(chunks))
        if self.neo4j_down:
            raise ConnectionError("Neo4j unavailable")
        return None


def _pipeline(monkeypatch):
    config = load_config().model_copy(deep=True)
    config.graph.enabled = True
    config.graph.extraction.enabled = True
    config.graph.extraction.llm_extraction_enabled = False
    extraction = _Extraction()
    monkeypatch.setattr(pipeline_module, "extract_and_write_for_document", extraction)
    store = _VectorStore()
    pipeline = IngestionPipeline(
        config, vectorstore=store, embedder=_Embedder(), graph_client=_GraphClient()
    )
    return pipeline, store, extraction


def test_failed_graph_write_is_retried_on_the_next_unchanged_ingest(tmp_path: Path, monkeypatch):
    """Neo4j down at first ingest; the identical re-ingest repairs the projection."""
    doc = tmp_path / "doc.md"
    doc.write_text("Document Service depends on PostgreSQL.", encoding="utf-8")
    pipeline, store, extraction = _pipeline(monkeypatch)

    extraction.neo4j_down = True
    first = pipeline.ingest_file(doc, "ds")
    assert first["changed"] is True
    assert store.get_graph_checksum(first["document_id"]) is None

    extraction.neo4j_down = False
    second = pipeline.ingest_file(doc, "ds")
    assert second["changed"] is False
    assert second["chunks_written"] == 0
    assert len(extraction.calls) == 2
    assert extraction.calls[1] > 0  # repaired from the stored chunks
    assert store.get_graph_checksum(first["document_id"]) is not None


def test_current_projection_is_not_rebuilt(tmp_path: Path, monkeypatch):
    """Once the graph checksum matches, an unchanged re-ingest does no graph work."""
    doc = tmp_path / "doc.md"
    doc.write_text("Document Service depends on PostgreSQL.", encoding="utf-8")
    pipeline, _, extraction = _pipeline(monkeypatch)

    pipeline.ingest_file(doc, "ds")
    pipeline.ingest_file(doc, "ds")

    assert len(extraction.calls) == 1


def test_graph_only_reset_is_rebuilt_on_the_next_ingest(tmp_path: Path, monkeypatch):
    """`reset_graph` clears the recorded projection, so re-ingest rebuilds it (GRAPH-004)."""
    doc = tmp_path / "doc.md"
    doc.write_text("Document Service depends on PostgreSQL.", encoding="utf-8")
    pipeline, _, extraction = _pipeline(monkeypatch)
    monkeypatch.setattr(pipeline._graph_writer, "delete_facts_by_dataset", lambda ds: {})

    pipeline.ingest_file(doc, "ds")
    pipeline.reset_graph("ds")
    result = pipeline.ingest_file(doc, "ds")

    assert result["changed"] is False
    assert len(extraction.calls) == 2
