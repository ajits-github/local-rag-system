from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline


def test_query_pipeline_answers_from_ingested_content(
    require_postgres, require_ollama, config, tmp_path: Path
):
    ingestion = IngestionPipeline(config)
    path = tmp_path / f"doc-{uuid.uuid4()}.txt"
    path.write_text(
        "The local RAG system stores vectors in Postgres using the pgvector "
        "extension, running on host port 15987.",
        encoding="utf-8",
    )
    ingest_result = ingestion.ingest_file(path)

    retrieval = RetrievalPipeline(config)
    try:
        result = retrieval.answer("What extension does the vector database use?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert isinstance(result["sources"], list)
    finally:
        ingestion._vectorstore.delete_chunks_by_document_id(ingest_result["document_id"])
