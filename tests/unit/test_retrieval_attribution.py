from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.eval.gold_schema import GoldExample, reference_context_bucket
from rag.eval.retrieval_attribution import (
    classify_contribution,
    classify_rrf_impact,
    evaluate_attribution,
    first_relevant_rank,
    select_interesting_examples,
)
from rag.retrieval.pipeline import RetrievalPipeline
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


# . classify_contribution ----------------------------------------------------


def test_classify_contribution_both_success():
    """Both retrievers finding evidence classifies as both_success."""
    assert classify_contribution(dense_hit=True, bm25_hit=True) == "both_success"


def test_classify_contribution_dense_only():
    """Only dense finding evidence classifies as dense_only_success."""
    assert classify_contribution(dense_hit=True, bm25_hit=False) == "dense_only_success"


def test_classify_contribution_bm25_only():
    """Only BM25 finding evidence classifies as bm25_only_success."""
    assert classify_contribution(dense_hit=False, bm25_hit=True) == "bm25_only_success"


def test_classify_contribution_neither():
    """Neither retriever finding evidence classifies as neither_success."""
    assert classify_contribution(dense_hit=False, bm25_hit=False) == "neither_success"


# . classify_rrf_impact -------------------------------------------------------


def test_rrf_impact_not_applicable_when_neither_retriever_found_it():
    """Neither retriever found it. fusion has nothing to have rescued."""
    assert classify_rrf_impact(None, None, None) == "not_applicable"


def test_rrf_impact_rescued_when_one_retriever_missed_but_fusion_found_it():
    """BM25 missed entirely; dense found it; fusion still found it. rescued."""
    assert classify_rrf_impact(dense_rank=3, bm25_rank=None, fused_rank=2) == "rescued"


def test_rrf_impact_rescued_symmetric_for_dense_missing():
    """Dense missed entirely; BM25 found it; fusion still found it. rescued."""
    assert classify_rrf_impact(dense_rank=None, bm25_rank=4, fused_rank=3) == "rescued"


def test_rrf_impact_still_missed_when_one_retriever_missed_and_fusion_also_missed():
    """One retriever missing rank is handled correctly when fusion also misses."""
    assert classify_rrf_impact(dense_rank=None, bm25_rank=5, fused_rank=None) == "still_missed"


def test_rrf_impact_improved_when_fused_rank_beats_best_single():
    """Both retrievers found it; fusion ranks it strictly better than either alone."""
    assert classify_rrf_impact(dense_rank=3, bm25_rank=4, fused_rank=1) == "improved"


def test_rrf_impact_unchanged_when_fused_rank_equals_best_single():
    """Both retrievers found it; fusion doesn't move the best available rank."""
    assert classify_rrf_impact(dense_rank=2, bm25_rank=5, fused_rank=2) == "unchanged"


def test_rrf_impact_degraded_when_fused_rank_worse_than_best_single():
    """Both retrievers found it; fusion ranks it worse than the better of the two."""
    assert classify_rrf_impact(dense_rank=1, bm25_rank=3, fused_rank=4) == "degraded"


def test_rrf_impact_degraded_when_fusion_drops_it_entirely():
    """Both retrievers found it individually, but fusion's own ranking misses it."""
    assert classify_rrf_impact(dense_rank=2, bm25_rank=3, fused_rank=None) == "degraded"


# . first_relevant_rank --------------------------------------------------------


def test_first_relevant_rank_finds_first_match():
    """Returns the 1-based rank of the first matching source."""
    sources = ["a.md", "knowledge_base/b.md", "c.md"]
    assert first_relevant_rank(sources, ["b.md"]) == 2


def test_first_relevant_rank_none_when_no_match():
    """Returns None when nothing in sources matches any relevant document."""
    assert first_relevant_rank(["a.md", "b.md"], ["z.md"]) is None


# . reference_context_bucket (gold_schema.py, shared with run_eval.py) --------


def test_reference_context_bucket_none_when_no_relevant_documents():
    """No relevant_documents at all. nothing to classify."""
    assert reference_context_bucket(["a.md"], ["some content"], [], ["excerpt"]) is None


def test_reference_context_bucket_c_when_document_missed():
    """The relevant document itself never appears in the retrieved sources."""
    bucket = reference_context_bucket(["other.md"], ["unrelated"], ["target.md"], ["excerpt"])
    assert bucket == "C"


def test_reference_context_bucket_not_applicable_when_no_reference_contexts_authored():
    """Document retrieved, but the gold example has no reference_contexts to check."""
    bucket = reference_context_bucket(["target.md"], ["any content"], ["target.md"], [])
    assert bucket == "not_applicable"


def test_reference_context_bucket_a_when_context_found():
    """Document retrieved and the supporting excerpt is present in its content."""
    bucket = reference_context_bucket(
        ["target.md"], ["the quick brown fox"], ["target.md"], ["quick brown"]
    )
    assert bucket == "A"


def test_reference_context_bucket_b_when_context_missing():
    """Document retrieved, but the supporting excerpt is not present in its content."""
    bucket = reference_context_bucket(
        ["target.md"], ["completely unrelated text"], ["target.md"], ["quick brown"]
    )
    assert bucket == "B"


# . select_interesting_examples ------------------------------------------------


def test_select_interesting_examples_prioritizes_rescued_then_degraded():
    """Rescued examples are preferred over degraded, which are preferred over improved."""
    per_example = [
        {"question": "q1", "rrf_impact": "unchanged"},
        {"question": "q2", "rrf_impact": "improved"},
        {"question": "q3", "rrf_impact": "degraded"},
        {"question": "q4", "rrf_impact": "rescued"},
    ]
    selected = select_interesting_examples(per_example, limit=2)
    assert [e["question"] for e in selected] == ["q4", "q3"]


def test_select_interesting_examples_respects_limit():
    """Never returns more than `limit` entries."""
    per_example = [{"question": f"q{i}", "rrf_impact": "rescued"} for i in range(10)]
    assert len(select_interesting_examples(per_example, limit=3)) == 3


# . evaluate_attribution end-to-end (fake pipeline, no real I/O) --------------


class _ScriptedVectorStore:
    """VectorStore double returning per-query dense/BM25 results, keyed by query text.

    `search()` only receives the embedding, not the raw query text. this
    double relies on `_IdentityEmbedder.embed_query` passing the text
    through unchanged so `search()` can key off it exactly like
    `search_keyword()` (which receives the text directly) does.
    """

    def __init__(self, by_query: dict[str, tuple[list[SearchResult], list[SearchResult]]]) -> None:
        """Store the {question: (dense_results, bm25_results)} script."""
        self._by_query = by_query

    def search(self, query_embedding, top_k, filters=None, auth=None) -> list[SearchResult]:
        """Look up dense results by the "embedding" (really the raw query text)."""
        dense, _bm25 = self._by_query[query_embedding]
        return dense[:top_k]

    def search_keyword(self, query, top_k, filters=None, auth=None) -> list[SearchResult]:
        """Look up BM25 results by the raw query text."""
        _dense, bm25 = self._by_query[query]
        return bm25[:top_k]


class _IdentityEmbedder:
    """Embedder double that returns the raw query text as its own "embedding"."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Unused by attribution; not exercised by these tests."""
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> str:
        """Pass the query text straight through, for _ScriptedVectorStore's lookup."""
        return text


def _example(question: str, relevant_documents: list[str]) -> GoldExample:
    """Build a minimal GoldExample for attribution tests."""
    return GoldExample(question=question, relevant_documents=relevant_documents)


def test_evaluate_attribution_classifies_dense_only_bm25_only_and_both():
    """A 3-question run classifies each question's contribution bucket correctly."""
    dense_only = _make_result("d1", "Dense hit.", source="dense_doc.md", score=0.9)
    bm25_only = _make_result("b1", "BM25 hit.", source="bm25_doc.md", score=3.0)
    dense_irrelevant = _make_result("di", "Irrelevant.", source="other.md", score=0.5)
    bm25_irrelevant = _make_result("bi", "Irrelevant.", source="other2.md", score=1.0)
    both_dense = _make_result("both1", "Both hit (dense view).", source="both_doc.md", score=0.8)
    both_bm25 = _make_result("both2", "Both hit (bm25 view).", source="both_doc.md", score=2.5)

    by_query = {
        "dense question": ([dense_only], [bm25_irrelevant]),
        "bm25 question": ([dense_irrelevant], [bm25_only]),
        "both question": ([both_dense], [both_bm25]),
    }
    vectorstore = _ScriptedVectorStore(by_query)
    config = load_config().model_copy(deep=True)
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=_IdentityEmbedder())

    examples = [
        _example("dense question", ["dense_doc.md"]),
        _example("bm25 question", ["bm25_doc.md"]),
        _example("both question", ["both_doc.md"]),
    ]
    report = evaluate_attribution(pipeline, examples, dataset_id="test-dataset")

    buckets = {e["question"]: e["contribution_bucket"] for e in report["per_example"]}
    assert buckets["dense question"] == "dense_only_success"
    assert buckets["bm25 question"] == "bm25_only_success"
    assert buckets["both question"] == "both_success"
    assert report["contribution_buckets"] == {
        "dense_only_success": 1,
        "bm25_only_success": 1,
        "both_success": 1,
    }


def test_evaluate_attribution_metrics_by_retriever_use_correct_rankings():
    """metrics_by_retriever's dense/bm25/hybrid Recall@5 reflect each ranking independently."""
    dense_hit = _make_result("d1", "Dense hit.", source="target.md", score=0.9)
    bm25_miss = _make_result("b1", "Irrelevant.", source="other.md", score=1.0)

    by_query = {"q": ([dense_hit], [bm25_miss])}
    vectorstore = _ScriptedVectorStore(by_query)
    config = load_config().model_copy(deep=True)
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=_IdentityEmbedder())

    report = evaluate_attribution(
        pipeline, [_example("q", ["target.md"])], dataset_id="test-dataset"
    )

    assert report["metrics_by_retriever"]["dense"]["recall@5"] == 1.0
    assert report["metrics_by_retriever"]["bm25"]["recall@5"] == 0.0
    # hybrid/fused is a union of both branches, so it inherits dense's hit.
    assert report["metrics_by_retriever"]["hybrid"]["recall@5"] == 1.0


def test_evaluate_attribution_reference_context_recovered_per_retriever():
    """reference_context_recovered differs per retriever when only one surfaces the excerpt."""
    dense_hit = _make_result("d1", "the quick brown fox jumps", source="target.md", score=0.9)
    bm25_hit_wrong_text = _make_result("b1", "no excerpt here", source="target.md", score=1.0)

    by_query = {"q": ([dense_hit], [bm25_hit_wrong_text])}
    vectorstore = _ScriptedVectorStore(by_query)
    config = load_config().model_copy(deep=True)
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=_IdentityEmbedder())

    example = GoldExample(
        question="q", relevant_documents=["target.md"], reference_contexts=["quick brown"]
    )
    report = evaluate_attribution(pipeline, [example], dataset_id="test-dataset")

    recovered = report["per_example"][0]["reference_context_recovered"]
    assert recovered["dense"] is True
    assert recovered["bm25"] is False


def test_evaluate_attribution_never_raises_when_relationship_expansion_enabled():
    """Attribution is unaffected by relationship_expansion.enabled. no expansion call is made."""
    dense_hit = _make_result("d1", "Dense hit.", source="target.md", score=0.9)
    by_query = {"q": ([dense_hit], [])}
    vectorstore = _ScriptedVectorStore(by_query)
    config = load_config().model_copy(deep=True)
    config.retrieval.relationship_expansion.enabled = True
    pipeline = RetrievalPipeline(config, vectorstore=vectorstore, embedder=_IdentityEmbedder())

    report = evaluate_attribution(
        pipeline, [_example("q", ["target.md"])], dataset_id="test-dataset"
    )

    assert report["per_example"][0]["dense_rank"] == 1
