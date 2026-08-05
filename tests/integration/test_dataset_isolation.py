"""Proves the dataset_id isolation mechanism that keeps unrelated corpora from sharing results.

E.g. the real data/knowledge_base "techfusion" dataset and the
data/sample_docs "sample_docs" dataset never share retrieval results.

Self-contained by design: it ingests two small synthetic documents into
two synthetic namespaces rather than depending on the real datasets being
ingested (data/knowledge_base isn't even committed to the repo, so a test
that assumed it was present would fail on a fresh clone). Proving the
underlying filter mechanism is equivalent to, and more portable than,
asserting it against the specific real dataset names.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline


def test_dataset_filter_excludes_other_namespace_even_as_strong_distractor(
    require_postgres, config, tmp_path: Path
):
    """The dataset_id filter isolates retrieval even against a near-identical distractor."""
    ns_a = f"pytest-isolation-a-{uuid.uuid4()}"
    ns_b = f"pytest-isolation-b-{uuid.uuid4()}"

    pipeline = IngestionPipeline(config)

    # Deliberately near-identical content differing only in the fact value,
    # so that without a working filter both chunks would be top candidates
    # for the same query -- a weak distractor wouldn't actually prove the
    # filter is doing anything.
    path_a = tmp_path / "a.txt"
    path_a.write_text(
        "The secret launch code for the Aurora rocket is ALPHA-7734.",
        encoding="utf-8",
    )
    path_b = tmp_path / "b.txt"
    path_b.write_text(
        "The secret launch code for the Aurora rocket is BETA-9912.",
        encoding="utf-8",
    )

    result_a = pipeline.ingest_file(path_a, ns_a)
    result_b = pipeline.ingest_file(path_b, ns_b)

    retrieval = RetrievalPipeline(config)
    query = "What is the secret launch code for the Aurora rocket?"
    try:
        results_a = retrieval.retrieve(
            query, filters={"dataset_id": ns_a}, top_k=10, rerank_top_n=10
        )
        results_b = retrieval.retrieve(
            query, filters={"dataset_id": ns_b}, top_k=10, rerank_top_n=10
        )

        assert results_a, "expected at least one result from ns_a"
        assert results_b, "expected at least one result from ns_b"

        assert all(r.chunk.metadata.dataset_id == ns_a for r in results_a)
        assert all(r.chunk.metadata.dataset_id == ns_b for r in results_b)

        assert any("ALPHA-7734" in r.chunk.content for r in results_a)
        assert not any("BETA-9912" in r.chunk.content for r in results_a)

        assert any("BETA-9912" in r.chunk.content for r in results_b)
        assert not any("ALPHA-7734" in r.chunk.content for r in results_b)
    finally:
        pipeline._vectorstore.delete_document(result_a["document_id"])
        pipeline._vectorstore.delete_document(result_b["document_id"])
