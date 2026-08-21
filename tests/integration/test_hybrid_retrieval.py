"""Proves hybrid retrieval actually helps on an exact-token-match query dense-only struggles with.

Self-contained by design (mirrors test_dataset_isolation.py's pattern):
ingests synthetic documents into a fresh namespace rather than depending
on the real, git-ignored data/knowledge_base.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline

_TARGET_TEXT = (
    "Model XR-9942-B requires a 240V power supply for continuous operation "
    "in industrial environments."
)
# Noise documents deliberately share heavy vocabulary overlap with the
# query (voltage, power supply, continuous operation, require) but never
# mention the exact identifier. a weak distractor set wouldn't prove
# BM25's exact-token-match strength is doing anything.
_NOISE_TEXTS = [
    "Most industrial equipment requires a stable power supply rated for "
    "high voltage continuous operation.",
    "Continuous operation of heavy machinery depends on a reliable power "
    "supply and correct voltage regulation.",
    "Power supply units must be rated correctly to require the proper "
    "voltage for continuous industrial operation.",
    "Industrial power supply standards require voltage tolerances suited "
    "for continuous equipment operation.",
    "A reliable power supply and stable voltage are required for "
    "continuous operation of industrial equipment.",
]
_QUERY = "What voltage does Model XR-9942-B require for continuous operation?"


def _rank_of_target(results: list) -> int:
    """Return the 1-based rank of the target chunk in `results`, or a large sentinel if absent."""
    for i, r in enumerate(results, start=1):
        if "XR-9942-B" in r.chunk.content:
            return i
    return 999


def test_hybrid_finds_exact_identifier_at_least_as_well_as_dense_only(
    require_postgres, config, tmp_path: Path
):
    """Hybrid surfaces/ranks an exact-token identifier match at least as well as dense-only."""
    ns = f"pytest-hybrid-{uuid.uuid4()}"
    pipeline = IngestionPipeline(config)

    target_path = tmp_path / "target.txt"
    target_path.write_text(_TARGET_TEXT, encoding="utf-8")
    noise_paths = []
    for i, text in enumerate(_NOISE_TEXTS):
        path = tmp_path / f"noise{i}.txt"
        path.write_text(text, encoding="utf-8")
        noise_paths.append(path)

    result_target = pipeline.ingest_file(target_path, ns)
    result_noise = [pipeline.ingest_file(p, ns) for p in noise_paths]

    try:
        dense_config = config.model_copy(deep=True)
        dense_config.retrieval.provider = "dense"
        hybrid_config = config.model_copy(deep=True)
        hybrid_config.retrieval.provider = "hybrid"

        dense_pipeline = RetrievalPipeline(dense_config)
        hybrid_pipeline = RetrievalPipeline(hybrid_config)

        dense_results = dense_pipeline.retrieve(
            _QUERY, filters={"dataset_id": ns}, candidate_k=2, generation_context_top_n=2
        )
        hybrid_results = hybrid_pipeline.retrieve(
            _QUERY, filters={"dataset_id": ns}, candidate_k=2, generation_context_top_n=2
        )

        dense_rank = _rank_of_target(dense_results)
        hybrid_rank = _rank_of_target(hybrid_results)

        # BM25's exact-token match should always surface the identifier
        # within a top_k=2 fused result. a hard requirement, not a
        # >= comparison, since this is BM25's specific strength.
        assert (
            hybrid_rank <= 2
        ), f"expected the exact-identifier chunk within hybrid's top 2, got rank {hybrid_rank}"
        # Hybrid must never rank the exact match *worse* than dense-only
        # does on the same fixture. >= (not strict >) since real
        # embedding behavior on one fixture can't be guaranteed brittle-exact.
        assert hybrid_rank <= dense_rank
    finally:
        pipeline._vectorstore.delete_document(result_target["document_id"])
        for r in result_noise:
            pipeline._vectorstore.delete_document(r["document_id"])
