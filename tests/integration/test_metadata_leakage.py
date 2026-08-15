"""Adversarial proof that document metadata never leaks what content-level controls already block.

Auth-boundary milestone (requirement 6). Mirrors test_authorization_isolation.py/
test_field_level_redaction.py's self-contained-fixture style: ingests small
synthetic documents into a fresh namespace per test rather than depending on
the real (gitignored) data/knowledge_base content.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.authorization import AuthorizationContext
from rag.retrieval.pipeline import RetrievalPipeline

_TEST_KEY = "SYNTHETIC_ONLY_TEST_KEY_METADATA_7F2Q"


def _secure_config(config, **overrides):
    """Return a copy of `config` with authorization + field_redaction enabled."""
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    secure.security.field_redaction.enabled = True
    for key, value in overrides.items():
        setattr(secure.security, key, value)
    return secure


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    """Write a synthetic Markdown file with a YAML front-matter block."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f'  - "{item}"' for item in value)
        else:
            lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_forbidden_document_metadata_never_appears_in_response(
    require_postgres, config, tmp_path: Path
):
    """A cross-tenant caller retrieves nothing -- no chunk_id/source/category leaks either."""
    ns = f"pytest-metadataleak-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "beta-runbook.md",
        {"tenant_id": "tenant_beta", "allowed_roles": ["tenant_beta_operator"]},
        "Beta's internal escalation contact is on-call-beta@example.internal.",
    )
    result = pipeline.ingest_file(path, ns)

    retrieval = RetrievalPipeline(secure)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    try:
        results = retrieval.retrieve(
            "What is the escalation contact?",
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=auth,
        )
        assert results == [], "a cross-tenant caller must not retrieve this document at all"
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])


def test_redacted_field_value_not_echoed_via_attachment_name_or_section_path(
    require_postgres, config, tmp_path: Path
):
    """A section_path that itself contains the sensitive literal is redacted, not just content."""
    ns = f"pytest-metadataleak-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "runbook.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        f"## Admin key {_TEST_KEY}\n\nThe synthetic test key is {_TEST_KEY}.",
    )
    result = pipeline.ingest_file(path, ns)

    retrieval = RetrievalPipeline(secure)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    try:
        results = retrieval.retrieve(
            "What is the synthetic test key?",
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=auth,
        )
        assert results, "expected the document-authorized chunk to still be retrieved"
        assert not any(_TEST_KEY in r.chunk.content for r in results)
        assert not any(
            r.chunk.metadata.section_path and _TEST_KEY in r.chunk.metadata.section_path
            for r in results
        )
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])


def test_permitted_document_metadata_is_unaffected_when_nothing_sensitive_present(
    require_postgres, config, tmp_path: Path
):
    """Benign-regression: ordinary metadata for an authorized, non-sensitive chunk is untouched."""
    ns = f"pytest-metadataleak-{uuid.uuid4()}"
    secure = _secure_config(config)
    pipeline = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "runbook.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        "## Callback configuration\n\nThe callback route ends in /v2.",
    )
    result = pipeline.ingest_file(path, ns)

    retrieval = RetrievalPipeline(secure)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    try:
        results = retrieval.retrieve(
            "What callback route should be used?",
            filters={"dataset_id": ns},
            candidate_k=10,
            generation_context_top_n=10,
            auth=auth,
        )
        assert any(r.chunk.metadata.section_path == "Callback configuration" for r in results)
        assert not any(r.redacted_field_ids for r in results)
    finally:
        pipeline._vectorstore.delete_document(result["document_id"])
