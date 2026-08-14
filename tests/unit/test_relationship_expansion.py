from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.eval.metrics import mean_recall_at_k
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata, SearchResult


def _chunk(
    chunk_id: str,
    content: str,
    document_id: str = "doc-1",
    section_path: str | None = "Setup",
    chunk_index: int = 0,
    parent_chunk_id: str | None = None,
    source: str = "a.md",
) -> Chunk:
    """Build a Chunk with minimal-but-valid metadata for expansion tests."""
    now = datetime.now(UTC)
    return Chunk(
        id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            source=source,
            source_type="markdown",
            created_at=now,
            last_modified=now,
            chunk_index=chunk_index,
            dataset_id="test-dataset",
            section_path=section_path,
            parent_chunk_id=parent_chunk_id,
        ),
    )


def _result(chunk: Chunk, score: float = 0.9) -> SearchResult:
    """Wrap a Chunk in a directly-retrieved SearchResult."""
    return SearchResult(chunk=chunk, score=score)


class FakeVectorStore:
    """Minimal VectorStore double: fixed search results, scripted expansion lookups.

    Records every `get_chunks_by_ids`/`get_chunks_by_section` call so tests
    can assert expansion never crosses a document_id/dataset boundary.
    """

    def __init__(
        self,
        results: list[SearchResult],
        chunks_by_id: dict[str, Chunk] | None = None,
        sections: dict[tuple[str, str | None], list[Chunk]] | None = None,
    ) -> None:
        """Store fixed search results and scripted lookup tables."""
        self._results = results
        self._chunks_by_id = chunks_by_id or {}
        self._sections = sections or {}
        self.get_chunks_by_ids_calls: list[list[str]] = []
        self.get_chunks_by_section_calls: list[tuple[str, str | None]] = []

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
        """Return the fixed dense results, ignoring the embedding/filters."""
        return self._results[:top_k]

    def search_keyword(self, query, top_k, filters=None) -> list[SearchResult]:
        """Return no keyword results -- these tests only exercise dense retrieval."""
        return []

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Record the call and return whichever scripted chunks match."""
        self.get_chunks_by_ids_calls.append(list(chunk_ids))
        return [self._chunks_by_id[cid] for cid in chunk_ids if cid in self._chunks_by_id]

    def get_chunks_by_section(self, document_id: str, section_path: str | None) -> list[Chunk]:
        """Record the call and return the scripted, chunk_index-ordered section."""
        self.get_chunks_by_section_calls.append((document_id, section_path))
        return self._sections.get((document_id, section_path), [])

    def get_cached_image_description(self, image_checksum: str) -> str | None:
        """Unused by RetrievalPipeline; not exercised by these tests."""
        return None

    def cache_image_description(self, *args, **kwargs) -> None:
        """Unused by RetrievalPipeline; not exercised by these tests."""


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
    """LLM double, unused by retrieve()-only tests but required by the pipeline constructor."""

    def generate(self, prompt: str) -> str:
        """Return a fixed placeholder response."""
        return "unused"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def _config_with_expansion(**overrides) -> object:
    """Return `load_config()` with `retrieval.relationship_expansion` fields overridden."""
    config = load_config()
    expansion = config.retrieval.relationship_expansion.model_copy(update=overrides)
    retrieval = config.retrieval.model_copy(update={"relationship_expansion": expansion})
    return config.model_copy(update={"retrieval": retrieval})


def _pipeline(config, vectorstore: FakeVectorStore) -> RetrievalPipeline:
    """Build a RetrievalPipeline wired to fake dependencies."""
    return RetrievalPipeline(
        config,
        vectorstore=vectorstore,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )


def test_expansion_disabled_by_default_is_a_noop():
    """With relationship_expansion.enabled left at its default (False), no expansion happens."""
    child = _chunk("c1", "child content", parent_chunk_id="parent-1")
    vectorstore = FakeVectorStore(
        results=[_result(child)], chunks_by_id={"parent-1": _chunk("parent-1", "parent content")}
    )
    pipeline = _pipeline(load_config(), vectorstore)

    results = pipeline.retrieve("q")

    assert [r.origin for r in results] == ["retrieved"]
    assert vectorstore.get_chunks_by_ids_calls == []
    assert vectorstore.get_chunks_by_section_calls == []


def test_parent_expansion_appends_parent_with_expanded_origin():
    """A retrieved chunk with a parent_chunk_id gets that parent appended, origin='expanded'."""
    child = _chunk("c1", "child content", parent_chunk_id="parent-1")
    parent = _chunk("parent-1", "parent content")
    vectorstore = FakeVectorStore(
        results=[_result(child, score=0.7)], chunks_by_id={"parent-1": parent}
    )
    config = _config_with_expansion(enabled=True, include_parent=True, include_neighbors=False)

    results = _pipeline(config, vectorstore).retrieve("q")

    assert [r.chunk.id for r in results] == ["c1", "parent-1"]
    assert results[1].origin == "expanded"
    assert results[1].expanded_from == "c1"
    assert results[1].score == 0.7  # inherited from the originating result, not fabricated


def test_neighbor_expansion_includes_previous_and_next_by_chunk_index():
    """include_neighbors adds the immediate prev/next chunk within the same section."""
    prev_chunk = _chunk("c0", "prev", chunk_index=0)
    target = _chunk("c1", "target", chunk_index=1)
    next_chunk = _chunk("c2", "next", chunk_index=2)
    vectorstore = FakeVectorStore(
        results=[_result(target)],
        sections={("doc-1", "Setup"): [prev_chunk, target, next_chunk]},
    )
    config = _config_with_expansion(enabled=True, include_parent=False, include_neighbors=True)

    results = _pipeline(config, vectorstore).retrieve("q")

    expanded_ids = {r.chunk.id for r in results if r.origin == "expanded"}
    assert expanded_ids == {"c0", "c2"}
    assert vectorstore.get_chunks_by_section_calls == [("doc-1", "Setup")]


def test_expansion_dedupes_against_already_retrieved_chunks():
    """A candidate parent that's already in the retrieved set is not appended a second time."""
    already_retrieved = _chunk("parent-1", "already have this one")
    child = _chunk("c1", "child content", parent_chunk_id="parent-1")
    vectorstore = FakeVectorStore(
        results=[_result(child), _result(already_retrieved)],
        chunks_by_id={"parent-1": already_retrieved},
    )
    config = _config_with_expansion(enabled=True, include_parent=True, include_neighbors=False)

    results = _pipeline(config, vectorstore).retrieve("q")

    assert [r.chunk.id for r in results] == ["c1", "parent-1"]
    assert all(r.origin == "retrieved" for r in results)


def test_expansion_caps_additions_at_max_related_elements():
    """Per originating result, at most max_related_elements chunks are appended."""
    target = _chunk("c1", "target", chunk_index=1, parent_chunk_id="parent-1")
    parent = _chunk("parent-1", "parent")
    prev_chunk = _chunk("c0", "prev", chunk_index=0)
    next_chunk = _chunk("c2", "next", chunk_index=2)
    vectorstore = FakeVectorStore(
        results=[_result(target)],
        chunks_by_id={"parent-1": parent},
        sections={("doc-1", "Setup"): [prev_chunk, target, next_chunk]},
    )
    config = _config_with_expansion(
        enabled=True, include_parent=True, include_neighbors=True, max_related_elements=2
    )

    results = _pipeline(config, vectorstore).retrieve("q")

    expanded = [r for r in results if r.origin == "expanded"]
    assert len(expanded) == 2  # parent + one neighbor, not all 3 candidates


def test_expansion_never_looks_outside_the_originating_chunks_own_document():
    """Parent/neighbor lookups are always scoped to the originating chunk's own document_id.

    A document_id is never shared across datasets (see CLAUDE.md's
    "Document identity" section), so scoping lookups to it is what keeps
    expansion from ever crossing a dataset_id boundary.
    """
    child = _chunk("c1", "child", document_id="doc-1", parent_chunk_id="parent-1")
    parent = _chunk("parent-1", "parent", document_id="doc-1")
    vectorstore = FakeVectorStore(
        results=[_result(child)],
        chunks_by_id={"parent-1": parent},
        sections={("doc-1", "Setup"): [child]},
    )
    config = _config_with_expansion(enabled=True, include_parent=True, include_neighbors=True)

    _pipeline(config, vectorstore).retrieve("q")

    assert vectorstore.get_chunks_by_section_calls == [("doc-1", "Setup")]


def test_expansion_happens_after_generation_context_cutoff():
    """A result truncated away by generation_context_top_n is never expanded.

    Two directly-retrieved results each have their own parent; overriding
    generation_context_top_n=1 keeps only the first for generation, so
    expansion must only ever see (and add a parent for) that first result
    -- proving expansion runs on the already-truncated primary list, not
    the full candidate pool.
    """
    first = _chunk("c1", "first", parent_chunk_id="parent-1")
    second = _chunk("c2", "second", parent_chunk_id="parent-2")
    vectorstore = FakeVectorStore(
        results=[_result(first, score=0.9), _result(second, score=0.8)],
        chunks_by_id={
            "parent-1": _chunk("parent-1", "parent of first"),
            "parent-2": _chunk("parent-2", "parent of second"),
        },
    )
    config = _config_with_expansion(enabled=True, include_parent=True, include_neighbors=False)

    results = _pipeline(config, vectorstore).retrieve("q", generation_context_top_n=1)

    assert [r.chunk.id for r in results] == ["c1", "parent-1"]
    assert vectorstore.get_chunks_by_ids_calls == [["parent-1"]]


def test_expanded_chunks_do_not_contaminate_recall_at_k():
    """An origin='expanded' chunk appended after the primary cutoff never counts toward Recall@k.

    The parent chunk's source would match "relevant", but recall_at_k(k=1)
    only looks at the first `k` entries -- since the parent is appended
    after the single primary result, it must not inflate recall@1, only
    recall@2 (once the cutoff reaches its position).
    """
    child = _chunk("c1", "child content", parent_chunk_id="parent-1", source="child.md")
    parent = _chunk("parent-1", "parent content", source="parent.md")
    vectorstore = FakeVectorStore(results=[_result(child)], chunks_by_id={"parent-1": parent})
    config = _config_with_expansion(enabled=True, include_parent=True, include_neighbors=False)

    results = _pipeline(config, vectorstore).retrieve("q", generation_context_top_n=1)
    sources = [r.chunk.metadata.source for r in results]

    assert sources == ["child.md", "parent.md"]  # the parent IS present, appended after
    assert mean_recall_at_k([sources], [["parent.md"]], 1) == 0.0  # but doesn't count at k=1
    assert mean_recall_at_k([sources], [["parent.md"]], 2) == 1.0  # counts once k reaches it


def test_expansion_disabled_ignores_relationship_config_fields():
    """include_parent/include_neighbors/max_related_elements are irrelevant when enabled=False."""
    child = _chunk("c1", "child content", parent_chunk_id="parent-1")
    vectorstore = FakeVectorStore(
        results=[_result(child)], chunks_by_id={"parent-1": _chunk("parent-1", "parent")}
    )
    config = _config_with_expansion(enabled=False, include_parent=True, include_neighbors=True)

    results = _pipeline(config, vectorstore).retrieve("q")

    assert len(results) == 1
    assert vectorstore.get_chunks_by_ids_calls == []
