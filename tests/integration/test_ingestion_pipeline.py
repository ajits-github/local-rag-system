from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline

TEST_DATASET_ID = "pytest-integration"


def test_ingest_file_writes_chunks_and_is_idempotent(require_postgres, config, tmp_path: Path):
    """Re-ingesting an unchanged file is a no-op; an edited file gets re-chunked."""
    pipeline = IngestionPipeline(config)
    path = tmp_path / f"doc-{uuid.uuid4()}.txt"
    path.write_text("Local RAG systems combine retrieval with generation. " * 20, encoding="utf-8")

    result_1 = pipeline.ingest_file(path, TEST_DATASET_ID)
    assert result_1["changed"] is True
    assert result_1["chunks_written"] > 0

    result_2 = pipeline.ingest_file(path, TEST_DATASET_ID)
    assert result_2["changed"] is False
    assert result_2["document_id"] == result_1["document_id"]

    path.write_text("Updated content that differs from the original. " * 20, encoding="utf-8")
    result_3 = pipeline.ingest_file(path, TEST_DATASET_ID)
    assert result_3["changed"] is True
    assert result_3["document_id"] == result_1["document_id"]

    pipeline._vectorstore.delete_document(result_1["document_id"])
