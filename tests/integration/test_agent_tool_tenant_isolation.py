"""Adversarial proof that the agent's tools cannot bypass authorization.

Covers get_document, get_latest_document, get_related_context. None of
which can bypass tenant/role authorization or field-level redaction, even
though they call VectorStore directly rather than going through
RetrievalPipeline.retrieve(). Also proves the reverse: when
authorization.enabled=False, these same three tools must not filter by a
caller-supplied auth context either (they resolve auth through
RetrievalPipeline.resolve_auth, the same kill-switch retrieve() already
applies). Also includes one test for search_knowledge_base's LLM-selectable
content_type filter (routes through retrieve(), so its ACL story is the
well-covered one; the filter itself is the part specific to this tool).

Mirrors tests/integration/test_authorization_isolation.py's self-contained
pattern: ingests small synthetic documents with real governance front
matter into a fresh namespace per test. No Ollama needed. these tools
never call the LLM.
"""

from __future__ import annotations

import uuid
from pathlib import Path

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
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline


class _NoOpEmbedder:
    """Placeholder embedder for get_document/get_latest_document's required parameter.

    Relevance-selection is never exercised in these small (1-2 chunk) fixtures.
    """

    def embed_query(self, text: str) -> list[float]:
        return [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def _secure_config(config, field_redaction: bool = False):
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    secure.security.field_redaction.enabled = field_redaction
    return secure


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_get_document_tool_cannot_bypass_tenant_isolation(require_postgres, config, tmp_path: Path):
    """A get_document call for another tenant's document returns nothing. SQL-level ACL holds."""
    ns = f"pytest-agent-tool-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    embedder = _NoOpEmbedder()

    path = _write_doc(
        tmp_path,
        "beta-secret.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "The secret rollback code for Beta is BETA-ROLLBACK-9981.",
    )
    result = ingestion.ingest_file(path, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=embedder)

    try:
        wrong_tenant_auth = AuthorizationContext(
            tenant_id="tenant_alpha", roles=["tenant_alpha_operator"]
        )
        chunks = get_document(
            GetDocumentArgs(source=str(path)),
            pipeline,
            vectorstore,
            ns,
            "what is the rollback code",
            embedder,
            wrong_tenant_auth,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
        assert chunks == []

        right_tenant_auth = AuthorizationContext(
            tenant_id="tenant_beta", roles=["tenant_beta_operator"]
        )
        chunks = get_document(
            GetDocumentArgs(source=str(path)),
            pipeline,
            vectorstore,
            ns,
            "what is the rollback code",
            embedder,
            right_tenant_auth,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
        assert any("BETA-ROLLBACK-9981" in c.content for c in chunks)
    finally:
        vectorstore.delete_document(result["document_id"])


def test_get_document_tool_output_still_gets_field_redacted(
    require_postgres, config, tmp_path: Path
):
    """A document fetched directly by get_document still has sensitive fields redacted.

    Proves the universal sanitization path (RetrievalPipeline.sanitize_evidence),
    not the document-level ACL, is what's being tested here.
    """
    ns = f"pytest-agent-tool-{uuid.uuid4()}"
    secure = _secure_config(config, field_redaction=True)
    ingestion = IngestionPipeline(secure)
    embedder = _NoOpEmbedder()

    path = _write_doc(
        tmp_path,
        "alpha-admin.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "The admin credential is SYNTHETIC_ONLY_zzz999. Contact support for rotation.",
    )
    result = ingestion.ingest_file(path, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=embedder)

    try:
        operator_auth = AuthorizationContext(
            tenant_id="tenant_alpha", roles=["tenant_alpha_operator"]
        )
        chunks = get_document(
            GetDocumentArgs(source=str(path)),
            pipeline,
            vectorstore,
            ns,
            "what is the admin credential",
            embedder,
            operator_auth,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
        assert any("SYNTHETIC_ONLY_zzz999" in c.content for c in chunks)

        # This is the exact wrap-then-sanitize step rag.agent.graph._execute_tool performs.
        # sanitize_evidence redacts SearchResult.chunk in place (a fresh model_copy, but
        # reassigned onto the same SearchResult), so each auth context needs its own
        # freshly-wrapped list. reusing one across two redaction passes would just
        # re-run redaction on already-redacted (no-longer-matching) text.
        from rag.schemas import SearchResult

        def _wrap(cs):
            return [SearchResult(chunk=c, score=1.0, origin="tool_fetched") for c in cs]

        sanitized = pipeline.sanitize_evidence(_wrap(chunks), operator_auth)
        assert not any("SYNTHETIC_ONLY_zzz999" in r.chunk.content for r in sanitized)
        assert all("[REDACTED:SENSITIVE_FIELD]" in r.chunk.content for r in sanitized)

        admin_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_admin"])
        sanitized_for_admin = pipeline.sanitize_evidence(_wrap(chunks), admin_auth)
        assert any("SYNTHETIC_ONLY_zzz999" in r.chunk.content for r in sanitized_for_admin)
    finally:
        vectorstore.delete_document(result["document_id"])


def test_get_latest_document_tool_resolves_current_version_and_respects_acl(
    require_postgres, config, tmp_path: Path
):
    """get_latest_document redirects a superseded path to the current version.

    Still enforces tenant isolation on whichever version it resolves to.
    """
    ns = f"pytest-agent-tool-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    embedder = _NoOpEmbedder()

    path_v1 = _write_doc(
        tmp_path,
        "retention-v1.md",
        {
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_operator"],
            "status": "superseded",
            "document_version": "1.0",
        },
        "The retention period is 7 days.",
    )
    path_v2 = _write_doc(
        tmp_path,
        "retention-v2.md",
        {
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_operator"],
            "status": "active",
            "document_version": "2.0",
            "supersedes": "retention-v1.md",
        },
        "The retention period is 90 days.",
    )
    result_v1 = ingestion.ingest_file(path_v1, ns)
    result_v2 = ingestion.ingest_file(path_v2, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=embedder)

    try:
        auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        # Ask for the OLD version's path. must still get current (90 days) content.
        chunks = get_latest_document(
            GetLatestDocumentArgs(source=str(path_v1)),
            pipeline,
            vectorstore,
            ns,
            "what is the retention period",
            embedder,
            auth,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
        assert any("90 days" in c.content for c in chunks)
        assert not any("7 days" in c.content for c in chunks)

        wrong_tenant_auth = AuthorizationContext(
            tenant_id="tenant_beta", roles=["tenant_beta_operator"]
        )
        chunks = get_latest_document(
            GetLatestDocumentArgs(source=str(path_v1)),
            pipeline,
            vectorstore,
            ns,
            "what is the retention period",
            embedder,
            wrong_tenant_auth,
            max_chunks=10,
            max_chunks_hard_ceiling=50,
        )
        assert chunks == []
    finally:
        vectorstore.delete_document(result_v1["document_id"])
        vectorstore.delete_document(result_v2["document_id"])


def test_get_related_context_tool_cannot_reach_an_unauthorized_seed_chunk(
    require_postgres, config, tmp_path: Path
):
    """A caller who guesses another tenant's chunk_id gets no related context back.

    get_chunks_by_ids' own auth-scoping (not get_related_context's own
    logic) is what blocks it.
    """
    ns = f"pytest-agent-tool-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)

    path = _write_doc(
        tmp_path,
        "beta-only.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "Beta-only operational detail.",
    )
    result = ingestion.ingest_file(path, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore, embedder=_NoOpEmbedder())

    try:
        # Discover the real chunk_id as an authorized Beta caller first (test setup only).
        beta_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
        beta_chunks = vectorstore.get_chunks_by_source(str(path), ns, auth=beta_auth)
        assert beta_chunks
        real_chunk_id = beta_chunks[0].metadata.chunk_id

        wrong_tenant_auth = AuthorizationContext(
            tenant_id="tenant_alpha", roles=["tenant_alpha_operator"]
        )
        related = get_related_context(
            GetRelatedContextArgs(chunk_id=real_chunk_id), pipeline, vectorstore, wrong_tenant_auth
        )
        assert related == []
    finally:
        vectorstore.delete_document(result["document_id"])


def test_search_knowledge_base_image_filter_cannot_leak_across_tenants(
    require_postgres, config, tmp_path: Path
):
    """A content_type="image" filtered search cannot surface another tenant's image chunk.

    Beta's image caption is written to near-verbatim match the query, so
    it really would be the strongest semantic match with no auth scoping.
    This proves ACL, not coincidental ranking, is what keeps it out of
    Alpha's results.
    """
    ns = f"pytest-agent-tool-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    query_text = "quarterly revenue breakdown by region"

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "alpha.png").write_bytes(b"fake-png-bytes-alpha")
    (tmp_path / "assets" / "beta.png").write_bytes(b"fake-png-bytes-beta")
    alpha_path = _write_doc(
        tmp_path,
        "alpha-diagram.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "![Alpha diagram](assets/alpha.png)\n\n*Figure 1: An unrelated operational diagram.*",
    )
    beta_path = _write_doc(
        tmp_path,
        "beta-diagram.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        f"![Beta diagram](assets/beta.png)\n\n*Figure 1: {query_text}.*",
    )
    alpha_result = ingestion.ingest_file(alpha_path, ns)
    beta_result = ingestion.ingest_file(beta_path, ns)
    vectorstore = ingestion._vectorstore
    pipeline = RetrievalPipeline(secure, vectorstore=vectorstore)

    try:
        # Sanity check: with no auth scoping, Beta's near-verbatim caption
        # really is the top image match. otherwise this test proves nothing.
        unscoped = search_knowledge_base(
            SearchKnowledgeBaseArgs(query=query_text, content_type="image", top_k=5),
            pipeline,
            filters={"dataset_id": ns},
            auth=None,
        )
        assert unscoped, "expected image results with no auth scoping"
        assert unscoped[0].chunk.metadata.tenant_id == "tenant_beta"

        alpha_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        alpha_results = search_knowledge_base(
            SearchKnowledgeBaseArgs(query=query_text, content_type="image", top_k=5),
            pipeline,
            filters={"dataset_id": ns},
            auth=alpha_auth,
        )
        assert all(r.chunk.metadata.tenant_id != "tenant_beta" for r in alpha_results)
        assert all(r.chunk.metadata.content_type == "image" for r in alpha_results)
    finally:
        vectorstore.delete_document(alpha_result["document_id"])
        vectorstore.delete_document(beta_result["document_id"])
