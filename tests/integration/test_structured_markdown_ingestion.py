"""Proves requirement 7 of the structured-content-ingestion milestone end to end.

Self-contained by design (matching test_dataset_isolation.py's precedent):
ingests a small synthetic Markdown document -- never the real, git-ignored
data/knowledge_base -- into a fresh pytest-namespaced dataset_id, retrieves
against it through a real Postgres round trip, and cleans up afterward.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline

_DOC_TEXT = """# Platform Notes

## Capacity Table

Some intro prose about the capacity table below.

| Region | Max Connections |
|---|---|
| eu-central-1 | 500 |
| us-east-1 | 250 |

Some closing prose about the capacity table above.

## Retry Helper

Here is the retry helper used by the worker pool:

```python
def retry_transient(fn):
    return fn()
```

## Reference Configuration

The default routing configuration is:

```json
{"allowed_countries": ["DE", "FR", "NL"]}
```

## Usage Chart

```text
Q1 |####
Q2 |######
```

*Chart caption: usage grew steadily quarter over quarter.*
"""


def _ingest(config, tmp_path: Path) -> tuple[IngestionPipeline, str, str]:
    """Ingest `_DOC_TEXT` into a fresh dataset_id; return (pipeline, dataset_id, document_id)."""
    dataset_id = f"pytest-structured-{uuid.uuid4()}"
    doc_path = tmp_path / "platform-notes.md"
    doc_path.write_text(_DOC_TEXT, encoding="utf-8")

    pipeline = IngestionPipeline(config)
    result = pipeline.ingest_file(doc_path, dataset_id)
    return pipeline, dataset_id, result["document_id"]


def test_table_row_question_retrieves_table_chunk(require_postgres, config, tmp_path: Path):
    """A question about a specific table row retrieves the content_type='table' chunk."""
    pipeline, dataset_id, document_id = _ingest(config, tmp_path)
    retrieval = RetrievalPipeline(config)
    try:
        results = retrieval.retrieve(
            "How many max connections does eu-central-1 allow?",
            filters={"dataset_id": dataset_id},
            candidate_k=5,
            generation_context_top_n=5,
        )
        assert any(r.chunk.metadata.content_type == "table" for r in results)
        table_result = next(r for r in results if r.chunk.metadata.content_type == "table")
        assert table_result.chunk.metadata.table_headers == ["Region", "Max Connections"]
        assert "eu-central-1" in table_result.chunk.content
    finally:
        pipeline._vectorstore.delete_document(document_id)


def test_function_question_retrieves_code_chunk(require_postgres, config, tmp_path: Path):
    """A question about a specific function retrieves the content_type='code' chunk."""
    pipeline, dataset_id, document_id = _ingest(config, tmp_path)
    retrieval = RetrievalPipeline(config)
    try:
        results = retrieval.retrieve(
            "What does the retry_transient function do?",
            filters={"dataset_id": dataset_id},
            candidate_k=5,
            generation_context_top_n=5,
        )
        assert any(r.chunk.metadata.content_type == "code" for r in results)
        code_result = next(r for r in results if r.chunk.metadata.content_type == "code")
        assert code_result.chunk.metadata.code_language == "python"
        assert "retry_transient" in code_result.chunk.content
    finally:
        pipeline._vectorstore.delete_document(document_id)


def test_json_setting_question_retrieves_configuration_chunk(
    require_postgres, config, tmp_path: Path
):
    """A question about a JSON setting retrieves the content_type='configuration' chunk."""
    pipeline, dataset_id, document_id = _ingest(config, tmp_path)
    retrieval = RetrievalPipeline(config)
    try:
        results = retrieval.retrieve(
            "Which countries are allowed in the routing configuration?",
            filters={"dataset_id": dataset_id},
            candidate_k=5,
            generation_context_top_n=5,
        )
        assert any(r.chunk.metadata.content_type == "configuration" for r in results)
        config_result = next(r for r in results if r.chunk.metadata.content_type == "configuration")
        assert config_result.chunk.metadata.code_language == "json"
        assert "allowed_countries" in config_result.chunk.content
    finally:
        pipeline._vectorstore.delete_document(document_id)


def test_chart_value_question_retrieves_chart_caption_chunk(
    require_postgres, config, tmp_path: Path
):
    """A question about a chart value retrieves the content_type='chart' chunk with its caption."""
    pipeline, dataset_id, document_id = _ingest(config, tmp_path)
    retrieval = RetrievalPipeline(config)
    try:
        results = retrieval.retrieve(
            "What does the usage chart caption say about growth?",
            filters={"dataset_id": dataset_id},
            candidate_k=5,
            generation_context_top_n=5,
        )
        assert any(r.chunk.metadata.content_type == "chart" for r in results)
        chart_result = next(r for r in results if r.chunk.metadata.content_type == "chart")
        assert "usage grew steadily quarter over quarter" in chart_result.chunk.content
    finally:
        pipeline._vectorstore.delete_document(document_id)


def test_mixed_prose_and_table_preserve_section_context(require_postgres, config, tmp_path: Path):
    """Prose and the table under the same header keep the same section_path after a round trip."""
    pipeline, dataset_id, document_id = _ingest(config, tmp_path)
    retrieval = RetrievalPipeline(config)
    try:
        results = retrieval.retrieve(
            "capacity table",
            filters={"dataset_id": dataset_id},
            candidate_k=10,
            generation_context_top_n=10,
        )
        capacity_results = [
            r for r in results if r.chunk.metadata.section_path == "Platform Notes > Capacity Table"
        ]
        assert capacity_results
        content_types = {r.chunk.metadata.content_type for r in capacity_results}
        assert "table" in content_types
    finally:
        pipeline._vectorstore.delete_document(document_id)
