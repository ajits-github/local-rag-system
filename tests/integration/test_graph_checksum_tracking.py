"""`documents.graph_checksum` round-trips against a real Postgres (BUG-015).

CI-safe: one synthetic document in a disposable dataset; no Neo4j needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline


def test_graph_checksum_set_read_and_clear(require_postgres, config, tmp_path: Path):
    """Unset after a vector-only ingest, recorded by set, and cleared per dataset."""
    ns = f"pytest-graph-checksum-{uuid.uuid4()}"
    doc = tmp_path / "doc.md"
    doc.write_text("Document Service depends on PostgreSQL.", encoding="utf-8")
    pipeline = IngestionPipeline(config)
    store = pipeline._vectorstore
    document_id = pipeline.ingest_file(doc, ns)["document_id"]
    try:
        assert store.get_graph_checksum(document_id) is None
        store.set_graph_checksum(document_id, "abc")
        assert store.get_graph_checksum(document_id) == "abc"
        store.clear_graph_checksums(ns)
        assert store.get_graph_checksum(document_id) is None
    finally:
        store.delete_document(document_id)
