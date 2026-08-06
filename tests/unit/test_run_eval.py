from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.eval.gold_schema import GoldExample
from rag.eval.run_eval import evaluate
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

    def generate(self, prompt: str) -> str:
        """Return a fixed response, ignoring `prompt`."""
        return "fake answer"

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
