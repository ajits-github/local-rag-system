"""Unit tests for graph-fact revalidation against Postgres (`rag.graph.fusion`).

Regressions for GRAPH-008 and SEC-018: graph facts used to be authorized only
against the ACL copy in Neo4j, and the `SearchResult` built from a fact carried
no tenant/classification/trust/sensitive-field metadata, so the egress gate
could not block graph evidence from a hosted provider.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.config import load_config
from rag.graph.fusion import graph_facts_to_search_results
from rag.graph.pipeline import GraphRAGPipeline
from rag.graph.retrieval import GraphEntityRef, GraphFact
from rag.retrieval.egress_policy import apply_egress_policy
from rag.retrieval.pipeline import source_dict
from rag.schemas import ChunkMetadata


def _metadata(document_id: str, **overrides) -> ChunkMetadata:
    now = datetime.now(UTC)
    fields = dict(
        document_id=document_id,
        chunk_id=f"{document_id}_0",
        source=f"knowledge_base/{document_id}.md",
        source_type="markdown",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds",
    )
    fields.update(overrides)
    return ChunkMetadata(**fields)


class _FakeVectorStore:
    """Answers `get_document_governance` from a fixed set of still-visible documents."""

    def __init__(self, visible: dict[str, ChunkMetadata]) -> None:
        self._visible = visible
        self.calls: list[tuple[list[str], str]] = []

    def get_document_governance(self, document_ids, dataset_id, auth=None):
        self.calls.append((list(document_ids), dataset_id))
        return {d: self._visible[d] for d in document_ids if d in self._visible}


def _fact(document_id: str | None, text: str = "Document Service DEPENDS_ON PostgreSQL"):
    return GraphFact(
        text=text,
        fact_type="relationship",
        entities=[
            GraphEntityRef(canonical_id="svc", name="Document Service", entity_type="Service")
        ],
        source_document_id=document_id,
        source_chunk_id=f"{document_id}_3" if document_id else None,
        dataset_id="ds",
    )


def test_fact_carries_current_postgres_governance_metadata():
    """Tenant, roles, classification, trust and status come from the Postgres source."""
    store = _FakeVectorStore(
        {
            "doc-a": _metadata(
                "doc-a",
                tenant_id="t1",
                allowed_roles=["eng"],
                classification="confidential",
                trust_level="authoritative",
                status="active",
            )
        }
    )

    [result] = graph_facts_to_search_results(
        [_fact("doc-a")], vectorstore=store, dataset_id="ds", auth=None
    )
    meta = result.chunk.metadata

    assert result.origin == "graph"
    assert meta.source == "knowledge_base/doc-a.md"
    assert (meta.tenant_id, meta.allowed_roles) == ("t1", ["eng"])
    assert (meta.classification, meta.trust_level, meta.status) == (
        "confidential",
        "authoritative",
        "active",
    )


def test_fact_whose_source_is_gone_or_unauthorized_is_dropped(caplog):
    """Fail closed: a fact without a visible Postgres source never becomes evidence."""
    store = _FakeVectorStore({"doc-a": _metadata("doc-a")})
    facts = [_fact("doc-a"), _fact("doc-revoked"), _fact(None)]

    with caplog.at_level("INFO", logger="rag.audit"):
        results = graph_facts_to_search_results(
            facts, vectorstore=store, dataset_id="ds", auth=None
        )

    assert [r.chunk.metadata.document_id for r in results] == ["doc-a"]
    assert any("graph_facts_failed_revalidation" in r.getMessage() for r in caplog.records)


def test_graph_evidence_from_a_confidential_document_is_blocked_from_egress():
    """End to end with the egress gate: graph evidence no longer bypasses classification."""
    config = load_config().model_copy(deep=True)
    config.security.egress_policy.enabled = True
    store = _FakeVectorStore({"doc-a": _metadata("doc-a", classification="confidential")})

    [result] = graph_facts_to_search_results(
        [_fact("doc-a")], vectorstore=store, dataset_id="ds", auth=None
    )
    decision = apply_egress_policy(source_dict(result), config)

    assert decision.allowed is False
    assert decision.blocked_reason == "classification"


def test_sensitive_values_in_fact_text_are_tagged_for_redaction():
    """A fact's own text is scanned like a chunk, so redaction and egress see its secrets."""
    store = _FakeVectorStore({"doc-a": _metadata("doc-a")})
    fact = _fact("doc-a", text="Billing Service USES SYNTHETIC_INTERNAL_TOKEN_abc123")

    [result] = graph_facts_to_search_results([fact], vectorstore=store, dataset_id="ds", auth=None)

    assert result.chunk.metadata.sensitive_field_ids == ["synthetic_internal_token"]


class _FakeRetrievalPipeline:
    """Only what `GraphRAGPipeline`'s graph-seeded path touches."""

    def __init__(self, store: _FakeVectorStore) -> None:
        self.vectorstore = store
        self.retrieve_calls = 0

    def resolve_auth(self, auth, filters=None, *, versions=None):
        return auth

    def retrieve(self, query, filters=None, candidate_k=None, auth=None):
        self.retrieve_calls += 1
        return []

    def answer(self, query, filters=None, candidate_k=None, auth=None):
        return {"answer": "vector", "sources": []}


class _FakeRetriever:
    def resolve_entities(self, query, *, dataset_id, auth, limit):
        return [GraphEntityRef(canonical_id="svc", name="Document Service", entity_type="Service")]


class _FakeGraphClient:
    def __init__(self, document_ids: list[str]) -> None:
        self._document_ids = document_ids
        self.cypher: list[str] = []

    def run_read(self, cypher, params):
        self.cypher.append(cypher)
        return [{"document_id": d} for d in self._document_ids]


def test_graph_seeded_vector_ignores_linked_documents_the_caller_cannot_read():
    """Seed ids are filtered through Postgres; an unreadable linked document is never seeded."""
    config = load_config().model_copy(deep=True)
    config.graph.enabled = True
    store = _FakeVectorStore({})
    pipeline = GraphRAGPipeline(
        config,
        _FakeRetrievalPipeline(store),
        graph_client=_FakeGraphClient(["doc-restricted"]),
        graph_retriever=_FakeRetriever(),
    )

    result = pipeline.answer(
        "What does Document Service depend on?",
        filters={"dataset_id": "ds"},
        strategy_override="graph_seeded_vector",
    )

    assert store.calls == [(["doc-restricted"], "ds")]
    assert result["graph_strategy_executed"] == "graph_seeded_vector_no_linked_documents"


def test_graph_seeded_neighbor_query_is_deterministically_ordered():
    """The neighbor lookup orders before LIMIT so the seed set is reproducible."""
    config = load_config().model_copy(deep=True)
    config.graph.enabled = True
    client = _FakeGraphClient([])
    pipeline = GraphRAGPipeline(
        config,
        _FakeRetrievalPipeline(_FakeVectorStore({})),
        graph_client=client,
        graph_retriever=_FakeRetriever(),
    )

    pipeline.answer("q", filters={"dataset_id": "ds"}, strategy_override="graph_seeded_vector")

    assert "ORDER BY document_id LIMIT 20" in client.cypher[0]


def test_two_facts_from_one_chunk_get_distinct_ids_distinct_from_the_chunk():
    """Graph ids are fact-specific, so dedup never collapses sibling facts (CTX-016)."""
    store = _FakeVectorStore({"doc-a": _metadata("doc-a")})
    facts = [
        _fact("doc-a", text="Document Service DEPENDS_ON PostgreSQL"),
        _fact("doc-a", text="Document Service DEPENDS_ON Redis"),
    ]

    results = graph_facts_to_search_results(facts, vectorstore=store, dataset_id="ds", auth=None)
    ids = [r.chunk.metadata.chunk_id for r in results]

    assert len(set(ids)) == 2
    assert all(i.startswith("graph:doc-a_3:") for i in ids)
