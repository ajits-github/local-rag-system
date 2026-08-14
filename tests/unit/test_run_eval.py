from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.eval.gold_schema import GoldExample
from rag.eval.run_eval import evaluate
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _make_result(
    chunk_id: str,
    content: str,
    source: str,
    score: float,
    attachment_name: str | None = None,
    source_anchor: str | None = None,
    content_type: str | None = None,
) -> SearchResult:
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
        attachment_name=attachment_name,
        source_anchor=source_anchor,
        content_type=content_type,
    )
    return SearchResult(chunk=Chunk(id=chunk_id, content=content, metadata=metadata), score=score)


class FakeVectorStore:
    """Minimal VectorStore double returning a fixed set of search results."""

    def __init__(self, results: list[SearchResult]) -> None:
        """Store the fixed results this double's search() will return."""
        self._results = results

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True

    def get_or_create_document_id(self, source: str, checksum: str, dataset_id: str):
        """Unused by evaluate(); not exercised by these tests."""
        raise NotImplementedError

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Unused by evaluate(); not exercised by these tests."""

    def delete_document(self, document_id: str) -> None:
        """Unused by evaluate(); not exercised by these tests."""

    def delete_dataset(self, dataset_id: str) -> None:
        """Unused by evaluate(); not exercised by these tests."""

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Unused by evaluate(); not exercised by these tests."""

    def search(self, query_embedding, top_k, filters=None) -> list[SearchResult]:
        """Return the fixed results, ignoring the query embedding/filters."""
        return self._results[:top_k]


class FakeEmbedder:
    """Minimal Embedder double returning a fixed placeholder vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text; unused here."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]


class FakeReranker:
    """Identity reranker double: returns results unchanged, truncated to top_n."""

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        """Truncate results to top_n without reordering."""
        return results[:top_n]


class FakeLLM:
    """LLM double that always returns a fixed response."""

    def __init__(self, response: str = "fake answer") -> None:
        """Store the fixed response this double's generate() will return."""
        self._response = response

    def generate(self, prompt: str) -> str:
        """Return the fixed response, ignoring `prompt`."""
        return self._response

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def test_evaluate_per_example_includes_generation_sources_with_content():
    """per_example entries carry generation_sources (with chunk content) from answer()."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )
    examples = [GoldExample(question="What is alpha?", expected_answer="Alpha.")]

    report = evaluate(pipeline, examples, dataset_id="test-dataset", run_generation=True)

    entry = report["per_example"][0]
    assert "generation_sources" in entry
    assert entry["generation_sources"][0]["content"] == "Alpha content."
    assert entry["generation_sources"][0]["source"] == "a.md"


def test_evaluate_reports_latency_breakdown_ms_when_generation_runs():
    """latency_breakdown_ms aggregates per-stage means from answer()'s latency_breakdown_ms."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )
    examples = [GoldExample(question="What is alpha?", expected_answer="Alpha.")]

    report = evaluate(pipeline, examples, dataset_id="test-dataset", run_generation=True)

    breakdown = report["latency_breakdown_ms"]
    assert "embed_ms" in breakdown
    assert "dense_search_ms" in breakdown
    assert "rerank_ms" in breakdown
    assert "generation_ms" in breakdown
    assert all(isinstance(v, float) for k, v in breakdown.items() if k != "note")


def test_evaluate_omits_latency_breakdown_ms_when_generation_skipped():
    """latency_breakdown_ms is absent when run_generation=False (no answer() calls made)."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )
    examples = [GoldExample(question="What is alpha?", expected_answer="Alpha.")]

    report = evaluate(pipeline, examples, dataset_id="test-dataset", run_generation=False)

    assert "latency_breakdown_ms" not in report


def _pipeline(results: list[SearchResult], llm=None) -> RetrievalPipeline:
    """Build a RetrievalPipeline wired to fake dependencies, dense retrieval only."""
    return RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm or FakeLLM(),
    )


def test_content_type_breakdown_groups_by_authored_content_type():
    """content_type_breakdown buckets by the gold row's own content_type, not a derived one."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [
        GoldExample(question="q1", relevant_documents=["a.md"], content_type="image_only"),
        GoldExample(question="q2", relevant_documents=["a.md"], content_type="image_only"),
        GoldExample(question="q3", relevant_documents=["a.md"]),  # no content_type -> uncategorized
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    breakdown = report["content_type_breakdown"]
    assert breakdown["image_only"]["count"] == 2
    assert breakdown["uncategorized"]["count"] == 1


def test_reference_context_bucket_a_when_doc_and_context_both_found():
    """Bucket A: relevant document retrieved AND its reference_contexts text is present."""
    results = [_make_result("c1", "The retry lock lasts 600 seconds.", source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="q1",
            relevant_documents=["a.md"],
            reference_contexts=["The retry lock lasts 600 seconds."],
        )
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["reference_context_analysis"]["buckets"] == {"A": 1}
    assert report["reference_context_analysis"]["supporting_context_hit_rate"] == 1.0
    assert report["per_example"][0]["reference_context_bucket"] == "A"


def test_reference_context_bucket_b_when_doc_found_but_context_missing():
    """Bucket B: relevant document retrieved but the specific reference_contexts text isn't."""
    content = "Unrelated content about something else."
    results = [_make_result("c1", content, source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="q1",
            relevant_documents=["a.md"],
            reference_contexts=["The retry lock lasts 600 seconds."],
        )
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["reference_context_analysis"]["buckets"] == {"B": 1}
    assert report["reference_context_analysis"]["supporting_context_hit_rate"] == 0.0


def test_reference_context_bucket_c_when_document_missed_entirely():
    """Bucket C: the relevant document itself never appears among retrieved sources."""
    results = [_make_result("c1", "content", source="unrelated.md", score=0.9)]
    examples = [
        GoldExample(
            question="q1", relevant_documents=["a.md"], reference_contexts=["some evidence"]
        )
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["reference_context_analysis"]["buckets"] == {"C": 1}
    # C doesn't count toward the A/(A+B) rate at all.
    assert report["reference_context_analysis"]["supporting_context_hit_rate"] is None


def test_reference_context_not_applicable_when_no_authored_reference_contexts():
    """A gold row with relevant_documents but no reference_contexts is 'not_applicable', not B."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [GoldExample(question="q1", relevant_documents=["a.md"])]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["reference_context_analysis"]["buckets"] == {"not_applicable": 1}


def test_relevant_image_hit_rate_matches_resolved_asset_path():
    """relevant_image_hit_rate matches a retrieved chunk's resolved source_anchor path."""
    results = [
        _make_result(
            "c1",
            "image content",
            source="data/knowledge_base/operations/api-performance-review.md",
            score=0.9,
            attachment_name="api-latency-by-hour.png",
            source_anchor="images/api-latency-by-hour.png",
            content_type="image",
        )
    ]
    examples = [
        GoldExample(
            question="q1",
            relevant_images=["knowledge_base/operations/images/api-latency-by-hour.png"],
        )
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["relevant_image_hit_rate"]["hit_rate"] == 1.0
    assert report["per_example"][0]["relevant_image_hit"] is True


def test_relevant_image_hit_rate_absent_when_no_examples_have_relevant_images():
    """The relevant_image_hit_rate key is omitted entirely when it doesn't apply to any example."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [GoldExample(question="q1", relevant_documents=["a.md"])]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert "relevant_image_hit_rate" not in report


def test_vision_behavior_correct_refusal_for_unanswerable_requires_vision_question():
    """A refusal on a gold-unanswerable, requires_vision question is classified correct_refusal."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="What CPU temperature is shown?",
            relevant_documents=["a.md"],
            unanswerable=True,
            requires_vision=True,
        )
    ]
    llm = FakeLLM("The CPU temperature is not shown in the documentation or images provided.")

    report = evaluate(
        _pipeline(results, llm), examples, dataset_id="test-dataset", run_generation=True
    )

    assert report["vision_behavior_breakdown"]["counts"] == {"correct_refusal": 1}
    assert report["per_example"][0]["vision_behavior"] == "correct_refusal"


def test_vision_behavior_hallucinated_answer_for_unanswerable_requires_vision_question():
    """A confident non-refusal on a gold-unanswerable, requires_vision question is hallucinated."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="What CPU temperature is shown?",
            relevant_documents=["a.md"],
            unanswerable=True,
            requires_vision=True,
        )
    ]
    llm = FakeLLM("The CPU temperature shown is 45 degrees Celsius.")

    report = evaluate(
        _pipeline(results, llm), examples, dataset_id="test-dataset", run_generation=True
    )

    assert report["vision_behavior_breakdown"]["counts"] == {"hallucinated_answer": 1}


def test_vision_behavior_caption_leak_success_for_answerable_requires_vision_question():
    """A non-refusal answer that overlaps expected_answer is classified caption_leak_success."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="What scaling trend does the chart caption describe?",
            expected_answer="Throughput flattens after six workers.",
            relevant_documents=["a.md"],
            unanswerable=False,
            requires_vision=True,
            content_type="caption_answerable",
        )
    ]
    llm = FakeLLM("Throughput flattens after six workers.")

    report = evaluate(
        _pipeline(results, llm), examples, dataset_id="test-dataset", run_generation=True
    )

    assert report["vision_behavior_breakdown"]["counts"] == {"caption_leak_success": 1}


def test_relationship_expansion_contribution_rate_zero_when_expansion_disabled():
    """Rate is 0.0 (not omitted) for requires_relationship_expansion examples when expansion is off.

    default.yaml's relationship_expansion.enabled is False, so no result
    ever has origin='expanded' -- the rate should reflect that honestly
    rather than silently omitting the key.
    """
    results = [_make_result("c1", "unrelated content", source="a.md", score=0.9)]
    examples = [
        GoldExample(
            question="q1",
            relevant_documents=["a.md"],
            reference_contexts=["the specific supporting passage"],
            requires_relationship_expansion=True,
        )
    ]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert report["relationship_expansion_contribution_rate"]["rate"] == 0.0


def test_relationship_expansion_contribution_rate_absent_when_not_applicable():
    """The key is omitted when no example is requires_relationship_expansion=True."""
    results = [_make_result("c1", "content", source="a.md", score=0.9)]
    examples = [GoldExample(question="q1", relevant_documents=["a.md"])]

    report = evaluate(_pipeline(results), examples, dataset_id="test-dataset", run_generation=False)

    assert "relationship_expansion_contribution_rate" not in report
