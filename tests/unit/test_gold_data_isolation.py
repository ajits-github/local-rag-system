"""Safeguards proving reference_contexts/reference_visual_contexts never reach ingestion/generation.

See gold_schema.py's GoldExample docstring: these fields are evaluation-only
ground truth, read exclusively by eval/*.py. Ingestion/retrieval/generation
never import rag.eval, and gold data flows one-directionally from eval code
-- it's never written into the chunks table and never handed to the LLM.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from rag.config import load_config
from rag.eval.gold_schema import GoldExample
from rag.eval.run_eval import evaluate
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata, SearchResult

_EVAL_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+rag\.eval\b", re.MULTILINE)
_GUARDED_PACKAGES = ("ingestion", "retrieval", "generation")


def test_ingestion_retrieval_generation_never_import_eval_modules():
    """Static guarantee: no module under ingestion/retrieval/generation imports rag.eval.*."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "rag"
    offenders = []
    for package in _GUARDED_PACKAGES:
        for path in (src_root / package).rglob("*.py"):
            if _EVAL_IMPORT_RE.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path))
    assert offenders == []


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

    def search(self, query_embedding, top_k, filters=None, auth=None) -> list[SearchResult]:
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


class RecordingFakeLLM:
    """LLM double that records every prompt it's ever called with, not just the last one."""

    def __init__(self) -> None:
        """Start with no recorded prompts."""
        self.prompts: list[str] = []

    def generate(self, system: str, user: str) -> str:
        """Record the combined system+user text and return a fixed, unrelated response."""
        self.prompts.append(f"{system}\n\n{user}" if system else user)
        return "The knowledge base does not contain that information."

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


def test_reference_visual_contexts_never_appears_in_a_generated_prompt():
    """A distinctive reference_visual_contexts marker never leaks into RetrievalPipeline's prompt.

    Ground truth flows into `evaluate()` only to be *compared against*
    retrieval/generation output after the fact -- it's never assembled
    into the context handed to the LLM, since `_build_context` only ever
    reads from `SearchResult`s sourced from the vector store.
    """
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id="c1",
        source="a.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
    )
    results = [
        SearchResult(
            chunk=Chunk(id="c1", content="Ordinary indexed content.", metadata=metadata), score=0.9
        )
    ]
    llm = RecordingFakeLLM()
    pipeline = RetrievalPipeline(
        load_config(),
        vectorstore=FakeVectorStore(results),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=llm,
    )
    marker = "SECRET_VISUAL_GROUND_TRUTH_MARKER_9f3a"
    examples = [
        GoldExample(
            question="What P95 latency is shown?",
            expected_answer="420 ms.",
            relevant_documents=["a.md"],
            requires_vision=True,
            reference_visual_contexts=[marker],
        )
    ]

    evaluate(pipeline, examples, dataset_id="test-dataset", run_generation=True)

    assert llm.prompts, "expected at least one generation call"
    assert all(marker not in prompt for prompt in llm.prompts)
