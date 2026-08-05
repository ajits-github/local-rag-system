"""Proves `VectorStore.delete_document` leaves zero rows in both documents and chunks.

Every other integration test relies on this teardown mechanism. Not just
`chunks`, which is what the old `delete_chunks_by_document_id`-based
teardown left orphaned (see PROJECT_JOURNAL.md for the incident this fixes).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg2

from rag.ingestion.pipeline import IngestionPipeline


def _row_counts(config, dataset_id: str) -> tuple[int, int]:
    """Return (documents row count, chunks row count) for dataset_id."""
    conn = psycopg2.connect(config.database_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM {config.vectorstore.documents_table} WHERE dataset_id = %s",
                (dataset_id,),
            )
            doc_count = cur.fetchone()[0]
            cur.execute(
                f"SELECT count(*) FROM {config.vectorstore.chunks_table} WHERE dataset_id = %s",
                (dataset_id,),
            )
            chunk_count = cur.fetchone()[0]
        return doc_count, chunk_count
    finally:
        conn.close()


def test_delete_document_leaves_no_document_or_chunk_rows(require_postgres, config, tmp_path: Path):
    """delete_document leaves zero rows in both documents and chunks tables."""
    dataset_id = f"pytest-teardown-{uuid.uuid4()}"
    pipeline = IngestionPipeline(config)
    path = tmp_path / "doc.txt"
    path.write_text(
        "Teardown hygiene test content, long enough to actually chunk. " * 10, encoding="utf-8"
    )

    result = pipeline.ingest_file(path, dataset_id)
    assert result["chunks_written"] > 0

    doc_count_before, chunk_count_before = _row_counts(config, dataset_id)
    assert doc_count_before == 1
    assert chunk_count_before == result["chunks_written"]

    pipeline._vectorstore.delete_document(result["document_id"])

    doc_count_after, chunk_count_after = _row_counts(config, dataset_id)
    assert doc_count_after == 0, "delete_document left an orphaned documents row"
    assert chunk_count_after == 0, "delete_document left orphaned chunk rows"
