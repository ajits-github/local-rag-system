"""Integration proof that authenticated-upload tenant governance actually holds end to end.

Self-contained by design (mirrors `test_authorization_isolation.py`): ingests
small synthetic documents into a fresh namespace per test via
`IngestionPipeline.ingest_file(..., caller=...)`, the same entrypoint
`POST /ingest` uses, then proves the persisted `tenant_id` and retrieval-
time authorization behave as the fix requires. `.txt` files stand in for
PDF/DOCX/HTML uploads: `TextLoader` never parses front matter for `.txt`
either, so it exercises the identical "loader produced no governance
metadata at all" code path without needing a binary fixture.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from rag.ingestion.governance import IngestCallerContext, IngestGovernanceError
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline


def _secure_config(config):
    """Return a copy of `config` with security.authorization.enabled=True."""
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    return secure


def test_pdf_like_upload_is_tenant_scoped_and_cross_tenant_is_denied(
    require_postgres, config, tmp_path: Path
):
    """A PDF/DOCX-equivalent upload with no governance metadata is stamped tenant_alpha.

    tenant_alpha can retrieve it, and tenant_beta cannot. [Tests A + B]
    """
    ns = f"pytest-ingest-governance-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)

    path = tmp_path / "customer-incident.txt"
    path.write_text(
        "The Aurora rocket's confidential incident code is INCIDENT-ALPHA-4471.",
        encoding="utf-8",
    )
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    result = pipeline.ingest_file(path, ns, caller=caller)

    retrieval = RetrievalPipeline(secure)
    query = "What is the Aurora rocket's confidential incident code?"
    try:
        alpha_auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
        alpha_results = retrieval.retrieve(
            query,
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=alpha_auth,
        )
        assert alpha_results, "tenant_alpha expected to retrieve its own uploaded document"
        assert all(r.chunk.metadata.tenant_id == "tenant_alpha" for r in alpha_results)

        beta_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
        beta_results = retrieval.retrieve(
            query,
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=beta_auth,
        )
        assert beta_results == [], "tenant_beta must never see tenant_alpha's uploaded document"
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])


def test_normal_caller_cross_tenant_front_matter_is_rejected_before_persistence(
    require_postgres, config, tmp_path: Path
):
    """A non-privileged caller's front-matter tenant_id override never reaches the DB. [Test E]."""
    ns = f"pytest-ingest-governance-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)

    path = tmp_path / "cross-tenant.md"
    path.write_text(
        '---\ntenant_id: "tenant_beta"\n---\n\nAttempted cross-tenant content.', encoding="utf-8"
    )
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)

    with pytest.raises(IngestGovernanceError):
        pipeline.ingest_file(path, ns, caller=caller)

    assert pipeline._vectorstore.list_document_sources(ns) == []


def test_privileged_caller_can_upload_on_behalf_of_another_tenant(
    require_postgres, config, tmp_path: Path
):
    """A cross_tenant_support_roles caller's explicit different tenant_id is honored.

    That tenant (not the uploader's own) can retrieve it. [Test F]
    """
    ns = f"pytest-ingest-governance-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)

    path = tmp_path / "support-authored.md"
    path.write_text(
        '---\ntenant_id: "tenant_beta"\n---\n\nSupport-authored content for tenant_beta.',
        encoding="utf-8",
    )
    caller = IngestCallerContext(tenant_id="techfusion_support", is_privileged=True)
    result = pipeline.ingest_file(path, ns, caller=caller)

    retrieval = RetrievalPipeline(secure)
    try:
        beta_auth = AuthorizationContext(tenant_id="tenant_beta", roles=["tenant_beta_operator"])
        beta_results = retrieval.retrieve(
            "support-authored content",
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=beta_auth,
        )
        assert beta_results
        assert all(r.chunk.metadata.tenant_id == "tenant_beta" for r in beta_results)
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])


def test_no_caller_preserves_pre_fix_ingestion_behavior(require_postgres, config, tmp_path: Path):
    """CLI-style ingestion (no caller) is completely unaffected by this fix. [Test G]."""
    ns = f"pytest-ingest-governance-{uuid.uuid4()}"
    pipeline = IngestionPipeline(config)

    path = tmp_path / "legacy.txt"
    path.write_text("Plain content ingested with no identity involved.", encoding="utf-8")
    result = pipeline.ingest_file(path, ns)  # caller defaults to None

    try:
        assert result["changed"] is True
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])
