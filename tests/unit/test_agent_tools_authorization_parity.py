"""Tests authorization parity between direct-fetch agent tools and search.

Agent tools calling VectorStore directly must respect
config.security.authorization.enabled the same way search_knowledge_base's
underlying RetrievalPipeline.retrieve() call already does.

These tests use a real RetrievalPipeline built with fakes (no Postgres) so
the actual resolve_auth logic is exercised, not a test double's
approximation of it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag.agent.tool_schemas import (
    GetDocumentArgs,
    GetLatestDocumentArgs,
    GetRelatedContextArgs,
    SearchKnowledgeBaseArgs,
)
from rag.agent.tools import (
    get_document,
    get_latest_document,
    get_related_context,
    search_knowledge_base,
)
from rag.config import load_config
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline
from rag.schemas import Chunk, ChunkMetadata


def _chunk(chunk_id: str, source: str = "policy.md", content: str = "content") -> Chunk:
    """Build a single Chunk with minimal-but-valid metadata."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        source=source,
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="ds1",
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


class RecordingVectorStore:
    """VectorStore double recording the `auth` each direct-fetch method actually received."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """Store the fixed chunks every fetch method on this double returns."""
        self._chunks = chunks
        self.get_chunks_by_source_calls: list[dict] = []
        self.get_chunks_by_ids_calls: list[dict] = []
        self.list_document_versions_calls: list[str] = []

    def list_document_versions(self, dataset_id: str):
        """Record the call and return no versions; freshness behavior is tested elsewhere."""
        self.list_document_versions_calls.append(dataset_id)
        return []

    def get_chunks_by_source(self, source, dataset_id, auth=None, limit=None):
        """Record the call and return the fixed chunks, ignoring source/dataset_id/limit."""
        self.get_chunks_by_source_calls.append(
            {"source": source, "dataset_id": dataset_id, "auth": auth}
        )
        return list(self._chunks)

    def get_chunks_by_ids(self, chunk_ids, auth=None):
        """Record the call and return the fixed chunks, ignoring chunk_ids."""
        self.get_chunks_by_ids_calls.append({"chunk_ids": chunk_ids, "auth": auth})
        return list(self._chunks)

    def get_chunks_by_section(self, document_id, section_path, auth=None):
        """No siblings; only the seed-fetch auth matters for these tests."""
        return []


class _NoOpEmbedder:
    """Placeholder embedder; relevance-selection is never exercised in these 1-chunk fixtures."""

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[0.0] for _ in texts]


def _config(authorization_enabled: bool):
    """Return `load_config()` with `security.authorization.enabled` set explicitly."""
    config = load_config().model_copy(deep=True)
    config.security.authorization.enabled = authorization_enabled
    return config


def test_get_document_ignores_caller_auth_when_authorization_disabled():
    """authorization.enabled=False: get_document's VectorStore call always receives auth=None."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(False), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_document(
        GetDocumentArgs(source="policy.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        _NoOpEmbedder(),
        caller_auth,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert vectorstore.get_chunks_by_source_calls[0]["auth"] is None


def test_get_latest_document_ignores_caller_auth_when_authorization_disabled():
    """authorization.enabled=False: get_latest_document's VectorStore call receives auth=None."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(False), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_latest_document(
        GetLatestDocumentArgs(source="policy.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        _NoOpEmbedder(),
        caller_auth,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert vectorstore.get_chunks_by_source_calls[0]["auth"] is None


def test_get_related_context_ignores_caller_auth_when_authorization_disabled():
    """authorization.enabled=False: get_related_context's seed fetch receives auth=None."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(False), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_related_context(
        GetRelatedContextArgs(chunk_id="doc-1_0"), pipeline, vectorstore, caller_auth
    )

    assert vectorstore.get_chunks_by_ids_calls[0]["auth"] is None


def test_search_knowledge_base_already_ignores_caller_auth_when_authorization_disabled():
    """search_knowledge_base is included so all four tools are covered.

    search_knowledge_base never needed a fix (retrieve() already resolves
    auth internally); this assertion documents parity with the direct
    fetch tools.
    """

    class SearchRecordingVectorStore(RecordingVectorStore):
        def __init__(self):
            super().__init__([])
            self.search_calls: list[dict] = []

        def search(self, query_embedding, top_k, filters=None, auth=None):
            self.search_calls.append({"auth": auth})
            return []

        def search_keyword(self, query, top_k, filters=None, auth=None):
            return []

    vectorstore = SearchRecordingVectorStore()
    pipeline = RetrievalPipeline(_config(False), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    search_knowledge_base(
        SearchKnowledgeBaseArgs(query="hello"), pipeline, {"dataset_id": "ds1"}, caller_auth
    )

    assert vectorstore.search_calls[0]["auth"] is None


def test_get_document_preserves_tenant_and_role_when_authorization_enabled():
    """authorization.enabled=True: get_document still enforces the caller's tenant/roles."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(True), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_document(
        GetDocumentArgs(source="policy.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        _NoOpEmbedder(),
        caller_auth,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    passed_auth = vectorstore.get_chunks_by_source_calls[0]["auth"]
    assert passed_auth is not None
    assert passed_auth.tenant_id == "tenant_alpha"
    assert passed_auth.roles == ["operator"]


def test_get_latest_document_preserves_tenant_and_role_when_authorization_enabled():
    """authorization.enabled=True: get_latest_document still enforces the caller's tenant/roles."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(True), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_latest_document(
        GetLatestDocumentArgs(source="policy.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        _NoOpEmbedder(),
        caller_auth,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    passed_auth = vectorstore.get_chunks_by_source_calls[0]["auth"]
    assert passed_auth is not None
    assert passed_auth.tenant_id == "tenant_alpha"
    assert passed_auth.roles == ["operator"]


def test_get_latest_document_fetches_list_document_versions_only_once():
    """get_latest_document reuses its own version fetch for resolve_auth's freshness resolution.

    get_latest_document needs the version list for source resolution, and
    resolve_auth can use that same list for freshness exclusions.
    """
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(True), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_latest_document(
        GetLatestDocumentArgs(source="policy.md"),
        pipeline,
        vectorstore,
        "ds1",
        "query",
        _NoOpEmbedder(),
        caller_auth,
        max_chunks=10,
        max_chunks_hard_ceiling=50,
    )

    assert vectorstore.list_document_versions_calls == ["ds1"]


def test_get_related_context_preserves_tenant_and_role_when_authorization_enabled():
    """authorization.enabled=True: get_related_context still enforces the caller's tenant/roles."""
    vectorstore = RecordingVectorStore([_chunk("doc-1_0")])
    pipeline = RetrievalPipeline(_config(True), vectorstore=vectorstore, embedder=_NoOpEmbedder())
    caller_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["operator"])

    get_related_context(
        GetRelatedContextArgs(chunk_id="doc-1_0"), pipeline, vectorstore, caller_auth
    )

    passed_auth = vectorstore.get_chunks_by_ids_calls[0]["auth"]
    assert passed_auth is not None
    assert passed_auth.tenant_id == "tenant_alpha"
    assert passed_auth.roles == ["operator"]


def test_no_tool_arg_model_accepts_an_auth_field():
    """No LLM-writable tool-arg schema exposes a field that could smuggle/override auth.

    `auth` is always a separate parameter from `args`, sourced only from
    AgentState.authorization_context (see rag.agent.tools' module
    docstring). Every arg model is extra="forbid", so an LLM response
    containing {"tenant_id": ..., "roles": ...} alongside a real
    argument is rejected outright by pydantic validation, never merged in.
    """
    import pydantic
    import pytest

    from rag.agent.tool_schemas import TOOL_ARG_MODELS

    for tool_name, model in TOOL_ARG_MODELS.items():
        assert "auth" not in model.model_fields
        assert "tenant_id" not in model.model_fields
        assert "roles" not in model.model_fields
        assert model.model_config.get("extra") == "forbid", tool_name

    with pytest.raises(pydantic.ValidationError):
        GetDocumentArgs(source="policy.md", tenant_id="tenant_beta")
    with pytest.raises(pydantic.ValidationError):
        GetRelatedContextArgs(chunk_id="doc-1_0", roles=["security_admin"])
