from __future__ import annotations

from datetime import datetime, timezone

from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.rerankers.noop import NoOpReranker
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _make_result(chunk_id: str, content: str, score: float) -> SearchResult:
    now = datetime.now(timezone.utc)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source="test.txt",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
    )
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=score)


def test_noop_reranker_truncates_to_top_n_without_reordering():
    results = [_make_result(f"c{i}", f"content {i}", score=1.0 - i * 0.1) for i in range(5)]

    reranked = NoOpReranker().rerank("any query", results, top_n=2)

    assert [r.chunk.id for r in reranked] == ["c0", "c1"]


def test_cross_encoder_reranker_reorders_by_predicted_score(monkeypatch):
    results = [
        _make_result("low", "irrelevant text", score=0.9),
        _make_result("high", "very relevant text", score=0.1),
    ]

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            # Score the second pair (originally lower vector-search score) higher.
            return [0.1, 0.9]

    monkeypatch.setattr("rag.rerankers.cross_encoder.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="fake-model")
    reranked = reranker.rerank("query", results, top_n=2)

    assert [r.chunk.id for r in reranked] == ["high", "low"]
    assert reranked[0].score == 0.9


def test_cross_encoder_reranker_handles_empty_results(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return []

    monkeypatch.setattr("rag.rerankers.cross_encoder.CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="fake-model")
    assert reranker.rerank("query", [], top_n=3) == []
