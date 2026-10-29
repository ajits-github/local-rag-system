"""`PgVectorStore.get_document_governance` against a real Postgres (GRAPH-008 revalidation).

CI-safe: ingests synthetic Markdown into a disposable dataset; no corpus,
Ollama, or Neo4j needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext


def _write_doc(tmp_path: Path, name: str, frontmatter: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


def test_governance_lookup_returns_current_metadata_and_respects_auth(
    require_postgres, config, tmp_path: Path
):
    """Visible documents resolve with governance fields; other tenants and deleted docs do not."""
    ns = f"pytest-governance-{uuid.uuid4()}"
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    pipeline = IngestionPipeline(secure)
    store = pipeline._vectorstore

    alpha = pipeline.ingest_file(
        _write_doc(
            tmp_path,
            "alpha.md",
            'tenant_id: "tenant_alpha"\n'
            'classification: "confidential"\n'
            'trust_level: "authoritative"',
            "Alpha Service depends on PostgreSQL.",
        ),
        ns,
    )
    beta = pipeline.ingest_file(
        _write_doc(
            tmp_path, "beta.md", 'tenant_id: "tenant_beta"', "Beta Service depends on Redis."
        ),
        ns,
    )
    alpha_id, beta_id = alpha["document_id"], beta["document_id"]
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=[])
    try:
        governance = store.get_document_governance([alpha_id, beta_id], ns, auth)

        assert set(governance) == {alpha_id}
        meta = governance[alpha_id]
        assert meta.tenant_id == "tenant_alpha"
        assert meta.classification == "confidential"
        assert meta.trust_level == "authoritative"
        assert meta.source.endswith("alpha.md")
        assert store.get_document_governance([alpha_id], "some-other-dataset", auth) == {}

        store.delete_document(alpha_id)
        assert store.get_document_governance([alpha_id], ns, auth) == {}
    finally:
        store.delete_document(alpha_id)
        store.delete_document(beta_id)
