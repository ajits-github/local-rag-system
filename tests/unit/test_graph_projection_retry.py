"""Durable graph-projection retry (BUG-015).

`documents.graph_checksum` records the last complete projection, and an
unchanged re-ingest repairs a missing or stale one. These tests keep the real
`extract_and_write_for_document` and fake only the external services, because
the real extractor returns normally on an outage rather than raising.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.graph.client import GraphUnavailableError
from rag.ingestion.pipeline import IngestionPipeline
from rag.schemas import Chunk


class _VectorStore:
    """Tracks document checksums, graph checksums, and stored chunks in memory."""

    def __init__(self) -> None:
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}
        self.graph_checksums: dict[str, str | None] = {}
        self.chunks: list[Chunk] = []

    def get_or_create_document_id(self, source, checksum, dataset_id):
        key = (source, dataset_id)
        if key not in self._documents:
            self._documents[key] = (f"doc-{len(self._documents) + 1}", "")
            return self._documents[key][0], True
        doc_id, existing = self._documents[key]
        return doc_id, existing != checksum

    def replace_document_chunks(self, document_id, checksum, chunks):
        key = next(k for k, v in self._documents.items() if v[0] == document_id)
        self._documents[key] = (document_id, checksum)
        self.chunks = list(chunks)

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        return [c for c in self.chunks if c.metadata.source == source]

    def get_graph_checksum(self, document_id):
        return self.graph_checksums.get(document_id)

    def set_graph_checksum(self, document_id, checksum):
        self.graph_checksums[document_id] = checksum

    def clear_graph_checksums(self, dataset_id):
        self.graph_checksums = {k: None for k in self.graph_checksums}

    def list_document_versions(self, dataset_id):
        return []


class _Embedder:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _GraphClient:
    """Fake `Neo4jClient`; the real writer and extractor run against it."""

    def __init__(self) -> None:
        self.healthy = True
        self.fail_write_matching: str | None = None
        self.writes: list[str] = []

    def health_check(self) -> bool:
        return self.healthy

    def run_write(self, cypher, params=None):
        if self.fail_write_matching and self.fail_write_matching in cypher:
            raise GraphUnavailableError("connection lost mid-write")
        self.writes.append(cypher)
        return {"relationships_deleted": 0, "nodes_deleted": 0}

    def projection_runs(self) -> int:
        return sum("n.first_seen_document_id = $document_id" in c for c in self.writes)


class _LLM:
    """Scripted extraction LLM: raises while `down`, else returns `response`."""

    def __init__(self, response: str) -> None:
        self.down = False
        self.response = response

    def generate(self, system, user):
        if self.down:
            raise ConnectionError("Ollama unavailable")
        return self.response


class _Template:
    def render(self, **_):
        return "system", "user"


_RELATIONSHIP_JSON = (
    '{"entities": [], "relationships": [{"subject_type": "Service", '
    '"subject_name": "Document Service", "relationship_type": "DEPENDS_ON", '
    '"object_type": "Database", "object_name": "PostgreSQL", "confidence": 0.9}]}'
)


def _pipeline(llm: _LLM | None = None):
    config = load_config().model_copy(deep=True)
    config.graph.enabled = True
    config.graph.extraction.enabled = True
    config.graph.extraction.llm_extraction_enabled = llm is not None
    store = _VectorStore()
    client = _GraphClient()
    pipeline = IngestionPipeline(
        config, vectorstore=store, embedder=_Embedder(), graph_client=client
    )
    if llm is not None:
        pipeline._graph_llm = llm
        pipeline._graph_extraction_template = _Template()
    return pipeline, store, client


def _doc(tmp_path: Path) -> Path:
    doc = tmp_path / "doc.md"
    doc.write_text("Document Service depends on PostgreSQL.", encoding="utf-8")
    return doc


def _assert_incomplete_then_repaired(pipeline, store, client, doc, restore) -> None:
    first = pipeline.ingest_file(doc, "ds")
    assert first["changed"] is True
    assert store.get_graph_checksum(first["document_id"]) is None

    runs_before = client.projection_runs()
    restore()
    second = pipeline.ingest_file(doc, "ds")
    assert second["changed"] is False
    assert second["chunks_written"] == 0
    assert client.projection_runs() == runs_before + 1
    assert store.get_graph_checksum(first["document_id"]) is not None


def test_failed_health_check_is_retried_on_the_next_unchanged_ingest(tmp_path: Path):
    """Neo4j down at first ingest: nothing is certified, and the identical re-ingest repairs it."""
    pipeline, store, client = _pipeline()
    client.healthy = False

    def restore():
        client.healthy = True

    _assert_incomplete_then_repaired(pipeline, store, client, _doc(tmp_path), restore)


def test_write_lost_during_deterministic_facts_is_retried(tmp_path: Path):
    """A connection lost after the health check leaves the projection uncertified."""
    pipeline, store, client = _pipeline()
    client.fail_write_matching = "MERGE (d:Document"

    def restore():
        client.fail_write_matching = None

    _assert_incomplete_then_repaired(pipeline, store, client, _doc(tmp_path), restore)


def test_failed_relationship_write_is_retried(tmp_path: Path):
    """Deterministic facts landing is not enough; a lost LLM relationship also blocks."""
    pipeline, store, client = _pipeline(_LLM(_RELATIONSHIP_JSON))
    client.fail_write_matching = "DEPENDS_ON"

    def restore():
        client.fail_write_matching = None

    _assert_incomplete_then_repaired(pipeline, store, client, _doc(tmp_path), restore)


def test_llm_outage_is_retried(tmp_path: Path):
    """An LLM outage must not certify a document as fully projected."""
    llm = _LLM(_RELATIONSHIP_JSON)
    llm.down = True
    pipeline, store, client = _pipeline(llm)

    def restore():
        llm.down = False

    _assert_incomplete_then_repaired(pipeline, store, client, _doc(tmp_path), restore)


def test_malformed_llm_output_does_not_block_the_projection(tmp_path: Path):
    """A chunk the model cannot parse would fail every retry, so it is not re-extracted forever."""
    pipeline, store, client = _pipeline(_LLM("not json"))
    doc = _doc(tmp_path)

    result = pipeline.ingest_file(doc, "ds")
    assert store.get_graph_checksum(result["document_id"]) is not None

    runs = client.projection_runs()
    pipeline.ingest_file(doc, "ds")
    assert client.projection_runs() == runs


def test_current_projection_is_not_rebuilt(tmp_path: Path):
    """Once the graph checksum matches, an unchanged re-ingest does no graph work."""
    pipeline, _, client = _pipeline()
    doc = _doc(tmp_path)

    pipeline.ingest_file(doc, "ds")
    pipeline.ingest_file(doc, "ds")

    assert client.projection_runs() == 1


def test_graph_only_reset_is_rebuilt_on_the_next_ingest(tmp_path: Path):
    """`reset_graph` clears the recorded projection, so re-ingest rebuilds it (GRAPH-004)."""
    pipeline, _, client = _pipeline()
    doc = _doc(tmp_path)

    pipeline.ingest_file(doc, "ds")
    pipeline.reset_graph("ds")
    result = pipeline.ingest_file(doc, "ds")

    assert result["changed"] is False
    assert client.projection_runs() == 2
