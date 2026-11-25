"""Version-comparison evaluation on the tracked synthetic fixture.

Ingests `data/version_history_demo_docs/` into a disposable dataset and runs
`rag.eval.run_eval.evaluate` over `data/eval/version_history_gold.jsonl` with
a deterministic word-hashing embedder, so the result does not depend on the
gitignored corpus, a downloaded model, or Ollama.
"""

from __future__ import annotations

import re
import uuid
import zlib
from pathlib import Path

import pytest

from rag.eval.gold_schema import load_gold_jsonl, source_matches_relevant
from rag.eval.run_eval import evaluate
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline

_REPO = Path(__file__).resolve().parents[2]
_DOCS = _REPO / "data" / "version_history_demo_docs"
_GOLD = _REPO / "data" / "eval" / "version_history_gold.jsonl"
_FUTURE_VERSION = "version_history_demo_docs/retention-policy-v3.md"


class _HashingEmbedder:
    """Normalized bag-of-words vector over hashed tokens; deterministic and offline."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def _embed(self, text: str) -> list[float]:
        """Hash each lowercase token into one dimension and L2-normalize."""
        vector = [0.0] * self._dimension
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % self._dimension] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each text."""
        return [self._embed(t) for t in texts]


class _Identity:
    """Reranker double that keeps order."""

    def rerank(self, query, results, top_n):
        """Return `results` truncated to `top_n`."""
        return results[:top_n]


@pytest.fixture
def fixture_eval(require_postgres, config):
    """Ingest the demo documents and return a retrieval pipeline and dataset id."""
    dataset_id = f"pytest-version-history-{uuid.uuid4()}"
    embedder = _HashingEmbedder(config.embedding.dimension)
    ingestion = IngestionPipeline(config, embedder=embedder, graph_client=None)
    ingestion.ingest_path(_DOCS, dataset_id=dataset_id)
    cfg = config.model_copy(deep=True)
    cfg.security.authorization.enabled = True
    cfg.retrieval.provider = "dense"
    cfg.retrieval.relationship_expansion.enabled = False
    pipeline = RetrievalPipeline(
        cfg, vectorstore=ingestion.vectorstore, embedder=embedder, reranker=_Identity()
    )
    try:
        yield pipeline, dataset_id, cfg
    finally:
        ingestion.clear_dataset(dataset_id)


def _found(retrieved: list[str], relevant: list[str]) -> list[bool]:
    """Return, per relevant path, whether any retrieved source matches it."""
    return [any(source_matches_relevant(s, r) for s in retrieved) for r in relevant]


def test_version_history_fixture_retrieves_every_compared_version(fixture_eval):
    """History rows retrieve each in-effect version; the current row retrieves only v2."""
    pipeline, dataset_id, cfg = fixture_eval
    examples = load_gold_jsonl(_GOLD)
    report = evaluate(pipeline, examples, dataset_id, run_generation=False, config=cfg)

    assert report["num_examples"] == 4
    assert report["retrieval"]["recall@10"] == pytest.approx(1.0)
    for row in report["per_example"]:
        assert all(_found(row["retrieved_sources"], row["relevant_documents"])), row["question"]
        # A version effective after query_as_of is never returned, even in history mode.
        retrieved = row["retrieved_sources"]
        assert not any(source_matches_relevant(s, _FUTURE_VERSION) for s in retrieved)

    safety = report["safety"]
    assert safety["document_unauthorized_retrieval_rate"]["rate"] == 0.0
    assert safety["stale_document_error_rate"]["rate"] == 0.0


def test_version_history_fixture_discriminates_current_mode(fixture_eval):
    """The same comparison rows in current mode lose v1, so the fixture tests the option."""
    pipeline, dataset_id, cfg = fixture_eval
    comparison = [
        e.model_copy(update={"version_scope": "current"})
        for e in load_gold_jsonl(_GOLD)
        if e.version_scope == "history" and len(e.relevant_documents) == 2
    ]
    report = evaluate(pipeline, comparison, dataset_id, run_generation=False, config=cfg)

    assert len(comparison) == 2
    for row in report["per_example"]:
        assert _found(row["retrieved_sources"], row["relevant_documents"]) == [False, True]
