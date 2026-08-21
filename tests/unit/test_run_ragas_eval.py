from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import load_config
from rag.eval import run_ragas_eval
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _make_result(chunk_id: str, content: str, source: str, score: float) -> SearchResult:
    """Build a SearchResult with minimal-but-valid chunk metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=score)


class _FakeVectorStore:
    """Duck-typed VectorStore double: only implements what retrieve() calls."""

    def __init__(self, results: list[SearchResult]) -> None:
        """Store the fixed results this double's search() will return."""
        self._results = results

    def search(self, query_embedding, top_k, filters=None, auth=None) -> list[SearchResult]:
        """Return the fixed results, ignoring the query embedding/filters."""
        return self._results[:top_k]


class _FakeEmbedder:
    """Duck-typed Embedder double."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


class _FakeReranker:
    """Identity reranker double: returns results unchanged, truncated to top_n."""

    def rerank(self, query, results, top_n) -> list[SearchResult]:
        """Truncate results to top_n without reordering."""
        return results[:top_n]


class _FakeLLM:
    """Duck-typed LLM double used for both generation and judging in these tests."""

    def __init__(self, response: str = "fake answer") -> None:
        """Store the fixed response this double's generate() will return."""
        self._response = response
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, system: str, user: str) -> str:
        """Return the fixed response, tracking call/token counts."""
        self.call_count += 1
        self.input_tokens += 10
        self.output_tokens += 5
        return self._response

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class _FakeCacheStats:
    """Duck-typed `CacheStats` double: fixed hits/misses."""

    def __init__(self, hits: int, misses: int) -> None:
        """Store the fixed hit/miss counts this double reports."""
        self.hits = hits
        self.misses = misses

    def as_dict(self) -> dict[str, int]:
        """Mirror `CacheStats.as_dict`."""
        return {"hits": self.hits, "misses": self.misses, "total": self.hits + self.misses}


class _FakeCache:
    """Duck-typed `NamespacedDiskCache` double: fixed stats, no real disk I/O."""

    def __init__(self, hits: int, misses: int) -> None:
        """Store fixed hit/miss counts as `.stats`."""
        self.stats = _FakeCacheStats(hits, misses)


def _write_gold(tmp_path: Path) -> Path:
    """Write a 3-question gold file: 2 with expected_answer, 1 without."""
    path = tmp_path / "gold.jsonl"
    lines = [
        '{"question": "What is pgvector?", "expected_answer": "A Postgres extension.", '
        '"relevant_documents": ["a.md"], "unanswerable": false}',
        '{"question": "What is the parental leave policy?", '
        '"expected_answer": "The knowledge base does not contain this policy.", '
        '"relevant_documents": [], "unanswerable": true}',
        '{"question": "No reference answer for this one", "relevant_documents": [], '
        '"unanswerable": false}',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fake_ragas_score(rows, judge_llm, embedder, cache=None) -> dict[str, Any]:
    """Canned ragas_scorer.score() replacement, recording what it was called with."""
    return {
        "metrics_used": ["faithfulness"],
        "metrics_failed": {},
        "aggregate": {"faithfulness": 0.8},
        "per_question": [
            {
                "question_index": r["question_index"],
                "question": r["question"],
                "unanswerable": r["unanswerable"],
                "scores": {"faithfulness": 0.8},
            }
            for r in rows
        ],
        "caveat": "test caveat",
        "cache_stats": cache.stats.as_dict() if cache is not None else None,
    }


def _patch_pipeline_backends(monkeypatch, results: list[SearchResult]) -> None:
    """Redirect RetrievalPipeline's internal factory calls to fakes."""
    monkeypatch.setattr(
        "rag.retrieval.pipeline.build_vectorstore", lambda config: _FakeVectorStore(results)
    )
    monkeypatch.setattr("rag.retrieval.pipeline.build_embedder", lambda config: _FakeEmbedder())
    monkeypatch.setattr("rag.retrieval.pipeline.build_reranker", lambda config: _FakeReranker())
    monkeypatch.setattr("rag.retrieval.pipeline.build_llm", lambda config: _FakeLLM())


def _patch_ragas_backends(monkeypatch, judge_llm: _FakeLLM, cache: Any = None) -> None:
    """Redirect run_ragas_eval's judge/embedder/scorer/cache calls to fakes.

    `cache` defaults to `None` (caching disabled) so these tests never
    touch the real on-disk RAGAS cache. matching `tests/unit`'s "no real
    I/O beyond what's mocked" convention. Pass a fake cache double to
    exercise the cache-enabled reporting path instead.
    """
    monkeypatch.setattr("rag.eval.run_ragas_eval.build_judge_llm", lambda config: judge_llm)
    monkeypatch.setattr("rag.eval.run_ragas_eval.build_embedder", lambda config: _FakeEmbedder())
    monkeypatch.setattr("rag.eval.run_ragas_eval.ragas_scorer.score", _fake_ragas_score)
    monkeypatch.setattr(
        "rag.eval.run_ragas_eval.ragas_cache.build_judge_cache", lambda config: cache
    )


def test_run_ragas_slices_examples_to_sample_size(tmp_path, monkeypatch):
    """Only the first --sample-size gold examples are run through the pipeline."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM())

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=2)

    assert report["num_examples"] == 2
    assert len(report["per_example"]) == 2


def test_run_ragas_skips_examples_missing_expected_answer(tmp_path, monkeypatch):
    """Examples with no expected_answer are excluded from ragas scoring and counted."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM())

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=3)

    assert report["ragas"]["num_scored"] == 2
    assert report["ragas"]["num_skipped_missing_expected_answer"] == 1


def test_run_ragas_report_includes_prompt_id_and_version(tmp_path, monkeypatch):
    """The ragas report section surfaces prompt_id/prompt_version for self-description."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM())
    config = load_config()

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    assert report["ragas"]["prompt_id"] == config.generation.prompt.id
    assert report["ragas"]["prompt_version"] == config.generation.prompt.version


def test_run_ragas_judge_summary_omits_api_usage_for_ollama_provider(tmp_path, monkeypatch):
    """estimated_api_usage is None when the judge provider is 'ollama' (no API cost)."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM())
    config = load_config().model_copy(deep=True)
    config.judge.provider = "ollama"
    monkeypatch.setattr("rag.eval.run_ragas_eval.load_config", lambda *a, **k: config)

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    assert report["ragas"]["judge"]["provider"] == "ollama"
    assert report["ragas"]["judge"]["estimated_api_usage"] is None


def test_run_ragas_judge_summary_includes_api_usage_for_hosted_provider(tmp_path, monkeypatch):
    """estimated_api_usage is populated from the judge LLM's own tracking for hosted providers."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    judge = _FakeLLM()
    _patch_ragas_backends(monkeypatch, judge)

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    usage = report["ragas"]["judge"]["estimated_api_usage"]
    assert usage is not None
    assert usage["call_count"] == judge.call_count


def test_run_ragas_reports_cache_disabled_when_no_cache_passed(tmp_path, monkeypatch):
    """report["ragas"]["cache"] is {"enabled": False} when caching is off."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM(), cache=None)

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    assert report["ragas"]["cache"] == {"enabled": False}


def test_run_ragas_reports_cache_hits_and_misses_when_enabled(tmp_path, monkeypatch):
    """report["ragas"]["cache"] surfaces hit/miss counts from an enabled cache."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM(), cache=_FakeCache(hits=3, misses=1))

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    cache_report = report["ragas"]["cache"]
    assert cache_report["enabled"] is True
    assert cache_report["hits"] == 3
    assert cache_report["misses"] == 1
    assert cache_report["total"] == 4


def test_run_ragas_avoided_cost_none_when_judge_never_called(tmp_path, monkeypatch):
    """avoided_cost_estimate explains itself instead of guessing when every call was a hit."""
    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    judge = _FakeLLM()  # call_count stays 0: nothing calls judge.generate() in this test
    _patch_ragas_backends(monkeypatch, judge, cache=_FakeCache(hits=5, misses=0))

    report = run_ragas_eval.run_ragas(gold_path, None, "test-dataset", sample_size=1)

    avoided = report["ragas"]["cache"]["avoided_cost_estimate"]
    assert avoided["estimated_cost_usd"] is None
    assert "reason" in avoided


def test_verbose_false_drops_per_example_and_per_question(tmp_path, monkeypatch, capsys):
    """--verbose (absent) truncates per_example and ragas.per_question from the printed report."""
    import json
    import sys

    gold_path = _write_gold(tmp_path)
    results = [_make_result("c1", "content", "a.md", 1.0)]
    _patch_pipeline_backends(monkeypatch, results)
    _patch_ragas_backends(monkeypatch, _FakeLLM())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ragas_eval", "--gold", str(gold_path), "--dataset-id", "test-dataset"],
    )

    run_ragas_eval.main()

    printed = json.loads(capsys.readouterr().out)
    assert "per_example" not in printed
    assert "per_question" not in printed["ragas"]
