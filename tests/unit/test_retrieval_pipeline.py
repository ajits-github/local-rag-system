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
    """Minimal VectorStore double returning fixed sets of search/search_keyword results.

    Records every `search`/`search_keyword` call (method name + kwargs)
    in `self.calls` so tests can assert exactly what `retrieve()` invoked.
    """

    def __init__(
        self, results: list[SearchResult], keyword_results: list[SearchResult] | None = None
    ) -> None:
        """Store the fixed results this double's search()/search_keyword() will return."""
        self._results = results
        self._keyword_results = keyword_results if keyword_results is not None else []
        self.calls: list[tuple[str, dict]] = []

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
        """Record the call and return the fixed dense results, ignoring the embedding."""
        self.calls.append(("search", {"top_k": top_k, "filters": filters}))
        return self._results[:top_k]

    def search_keyword(self, query, top_k, filters=None) -> list[SearchResult]:
        """Record the call and return the fixed keyword results, ignoring the query text."""
        self.calls.append(("search_keyword", {"top_k": top_k, "filters": filters}))
        return self._keyword_results[:top_k]


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


def test_answer_sources_include_chunk_content():
    """Each entry in answer()'s sources list includes the chunk's raw content."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    result = pipeline.answer("What is alpha?")

    assert result["sources"][0]["content"] == "Alpha content."


def test_retrieve_dense_provider_never_calls_search_keyword():
    """Default (dense) provider: retrieve() only calls search(), never search_keyword()."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    vectorstore = FakeVectorStore(results)
    config = load_config()
    assert config.retrieval.provider == "dense"
    pipeline = RetrievalPipeline(
        config, vectorstore=vectorstore, embedder=FakeEmbedder(), reranker=FakeReranker()
    )

    pipeline.retrieve("What is alpha?")

    methods_called = [call[0] for call in vectorstore.calls]
    assert methods_called == ["search"]


def test_retrieve_dense_provider_calls_search_with_configured_top_k():
    """Dense provider's search() call uses config.retrieval.top_k and passes filters through."""
    results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    vectorstore = FakeVectorStore(results)
    config = load_config()
    pipeline = RetrievalPipeline(
        config, vectorstore=vectorstore, embedder=FakeEmbedder(), reranker=FakeReranker()
    )

    pipeline.retrieve("What is alpha?", filters={"dataset_id": "techfusion"})

    assert vectorstore.calls == [
        ("search", {"top_k": config.retrieval.top_k, "filters": {"dataset_id": "techfusion"}})
    ]


def test_retrieve_hybrid_provider_calls_both_search_methods():
    """Hybrid provider: retrieve() calls both search() and search_keyword() with matching args."""
    dense_results = [_make_result("c1", "Alpha content.", source="a.md", score=0.9)]
    keyword_results = [_make_result("c2", "Beta content.", source="b.md", score=5.0)]
    vectorstore = FakeVectorStore(dense_results, keyword_results)
    config = load_config().model_copy(deep=True)
    config.retrieval.provider = "hybrid"
    pipeline = RetrievalPipeline(
        config, vectorstore=vectorstore, embedder=FakeEmbedder(), reranker=FakeReranker()
    )

    pipeline.retrieve("What is alpha?", filters={"dataset_id": "techfusion"})

    methods_called = [call[0] for call in vectorstore.calls]
    assert methods_called == ["search", "search_keyword"]
    for _method, kwargs in vectorstore.calls:
        assert kwargs == {"top_k": config.retrieval.top_k, "filters": {"dataset_id": "techfusion"}}


def test_retrieve_hybrid_provider_fuses_dense_and_keyword_results():
    """Hybrid provider fuses both branches via RRF before handing off to the reranker."""
    shared = _make_result("shared", "In both.", source="a.md", score=0.9)
    dense_only = _make_result("dense-only", "Dense only.", source="b.md", score=0.5)
    keyword_only = _make_result("keyword-only", "Keyword only.", source="c.md", score=3.0)
    vectorstore = FakeVectorStore(
        results=[shared, dense_only], keyword_results=[shared, keyword_only]
    )
    config = load_config().model_copy(deep=True)
    config.retrieval.provider = "hybrid"
    pipeline = RetrievalPipeline(
        config, vectorstore=vectorstore, embedder=FakeEmbedder(), reranker=FakeReranker()
    )

    fused = pipeline.retrieve("query", top_k=10, rerank_top_n=10)

    ids = [r.chunk.id for r in fused]
    assert set(ids) == {"shared", "dense-only", "keyword-only"}
    # "shared" is ranked 1st in both branches, so it must outrank either
    # single-branch result -- the direct signature of correct RRF fusion.
    assert ids[0] == "shared"


def test_retrieve_hybrid_uses_configured_rrf_k():
    """A different config.retrieval.hybrid.rrf_k threads through to the fused RRF score."""
    shared = _make_result("shared", "In both.", source="a.md", score=0.9)
    vectorstore = FakeVectorStore(results=[shared], keyword_results=[shared])
    config = load_config().model_copy(deep=True)
    config.retrieval.provider = "hybrid"
    config.retrieval.hybrid.rrf_k = 1
    pipeline = RetrievalPipeline(
        config, vectorstore=vectorstore, embedder=FakeEmbedder(), reranker=FakeReranker()
    )

    fused = pipeline.retrieve("query", top_k=5, rerank_top_n=5)

    assert fused[0].score == 2 * (1.0 / 2)  # k=1, rank=1 in both lists: 2 * 1/(1+1)
