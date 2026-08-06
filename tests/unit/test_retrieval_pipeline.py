from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.prompts.loader import PromptTemplate
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
        """Unused by RetrievalPipeline; not exercised by these tests."""
        raise NotImplementedError

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        """Unused by RetrievalPipeline; not exercised by these tests."""

    def delete_document(self, document_id: str) -> None:
        """Unused by RetrievalPipeline; not exercised by these tests."""

    def delete_dataset(self, dataset_id: str) -> None:
        """Unused by RetrievalPipeline; not exercised by these tests."""

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Unused by RetrievalPipeline; not exercised by these tests."""

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
    """LLM double that records the last prompt it was called with."""

    def __init__(self, response: str = "fake answer") -> None:
        """Store the fixed response this double's generate() will return."""
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        """Record `prompt` and return the fixed response."""
        self.last_prompt = prompt
        return self._response

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def test_answer_builds_labeled_context_and_uses_configured_prompt():
    """answer() labels each chunk with its source and renders the configured v1 prompt."""
    results = [
        _make_result("c1", "Alpha content.", source="a.md", score=0.9),
        _make_result("c2", "Beta content.", source="b.md", score=0.8),
    ]
    llm = FakeLLM()
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm,
    )

    pipeline.answer("What is alpha?")

    assert llm.last_prompt is not None
    assert "[Source 1: a.md]\nAlpha content." in llm.last_prompt
    assert "[Source 2: b.md]\nBeta content." in llm.last_prompt
    assert llm.last_prompt.startswith("Answer the question using only the context below.")


def test_pipeline_uses_injected_prompt_template_with_system_message():
    """A non-empty system_template is concatenated ahead of the rendered user message."""
    template = PromptTemplate(
        prompt_id="test",
        version="v-test",
        description="test",
        system_template="SYSTEM: {query}",
        user_template="USER: {context}",
        required_variables=["context", "query"],
        created_at="2026-08-06",
    )
    results = [_make_result("c1", "content", source="a.md", score=1.0)]
    llm = FakeLLM()
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm,
        prompt_template=template,
    )

    pipeline.answer("hi")

    assert llm.last_prompt == "SYSTEM: hi\n\nUSER: [Source 1: a.md]\ncontent"
