from __future__ import annotations

from datetime import UTC, datetime

from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.rerankers.noop import NoOpReranker
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _make_result(chunk_id: str, content: str, score: float) -> SearchResult:
    """Build a SearchResult with minimal-but-valid chunk metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source="test.txt",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=score)


def test_noop_reranker_is_a_true_identity():
    """NoOpReranker returns every result unchanged, ignoring top_n entirely. no truncation."""
    results = [_make_result(f"c{i}", f"content {i}", score=1.0 - i * 0.1) for i in range(5)]

    reranked = NoOpReranker().rerank("any query", results, top_n=2)

    assert reranked == results
    assert [r.chunk.id for r in reranked] == ["c0", "c1", "c2", "c3", "c4"]


def test_noop_reranker_does_not_rescore():
    """NoOpReranker never mutates a result's .score."""
    results = [_make_result("c0", "content", score=0.42)]

    reranked = NoOpReranker().rerank("any query", results, top_n=1)

    assert reranked[0].score == 0.42
    assert reranked[0] is results[0]


def test_cross_encoder_reranker_reorders_by_predicted_score(monkeypatch):
    """CrossEncoderReranker reorders results by the model's predicted score."""
    results = [
        _make_result("low", "irrelevant text", score=0.9),
        _make_result("high", "very relevant text", score=0.1),
    ]

    class FakeCrossEncoder:
        """Stand-in for sentence-transformers CrossEncoder."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Accept and ignore any CrossEncoder constructor args."""

        def predict(self, pairs):
            """Score the second pair (originally lower vector-search score) higher."""
            return [0.1, 0.9]

    monkeypatch.setattr("rag.rerankers.cross_encoder.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="fake-model")
    reranked = reranker.rerank("query", results, top_n=2)

    assert [r.chunk.id for r in reranked] == ["high", "low"]
    assert reranked[0].score == 0.9


def test_cross_encoder_reranker_handles_empty_results(monkeypatch):
    """CrossEncoderReranker returns an empty list unchanged, without predicting."""

    class FakeCrossEncoder:
        """Stand-in for sentence-transformers CrossEncoder."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Accept and ignore any CrossEncoder constructor args."""

        def predict(self, pairs):
            """Unused: rerank short-circuits before calling predict on empty input."""
            return []

    monkeypatch.setattr("rag.rerankers.cross_encoder.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="fake-model")
    assert reranker.rerank("query", [], top_n=3) == []
