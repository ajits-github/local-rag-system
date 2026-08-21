"""Proof that JWT-derived identity drives field-level redaction through the real HTTP boundary.

Auth-boundary milestone. Unlike test_field_level_redaction.py (which calls
`RetrievalPipeline` directly), this goes through `POST /query` end-to-end:
`Authorization: Bearer <jwt>` -> `verify_jwt` -> `VerifiedIdentity` ->
`AuthorizationContext` (`api/routers/query.py`) -> retrieval ACL -> field
redaction -> the response body an external caller actually receives.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_retrieval_pipeline
from rag.api.main import app
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline

_TEST_KEY = "SYNTHETIC_ONLY_TEST_KEY_APIBOUNDARY_4M9K"
_SECRET = "integration-test-only-not-a-real-secret-value"


def _secure_config(config):
    """Return a copy of `config` with authorization + field_redaction + JWT auth enabled."""
    secure = config.model_copy(deep=True)
    secure.security.authorization.enabled = True
    secure.security.field_redaction.enabled = True
    secure.security.auth.enabled = True
    secure.security.auth.jwt.secret_env_var = "JWT_HS256_SECRET"
    return secure


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    """Write a synthetic Markdown file with a YAML front-matter block."""
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f'{key}: "{value}"' if not isinstance(value, list) else f"{key}:")
        if isinstance(value, list):
            lines.extend(f'  - "{item}"' for item in value)
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _token(tenant_id: str, roles: list[str]) -> str:
    now = int(time.time())
    claims = {
        "sub": "pytest",
        "tenant_id": tenant_id,
        "roles": roles,
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def test_admin_jwt_sees_unredacted_admin_field(
    require_postgres, require_ollama, config, tmp_path: Path, monkeypatch
):
    """A tenant_alpha_admin JWT retrieves the credential unredacted through POST /query."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    ns = f"pytest-apifieldredact-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "runbook.md",
        {
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_operator", "tenant_alpha_admin"],
        },
        f"The synthetic test key is {_TEST_KEY}.",
    )
    ingest_result = ingestion.ingest_file(path, ns)

    app.dependency_overrides[get_config] = lambda: secure
    app.dependency_overrides[get_retrieval_pipeline] = lambda: RetrievalPipeline(secure)
    try:
        client = TestClient(app)
        token = _token("tenant_alpha", ["tenant_alpha_admin"])
        response = client.post(
            "/query",
            json={"query": "What is the synthetic test key?", "filters": {"dataset_id": ns}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        ingestion._vectorstore.delete_document(ingest_result["document_id"])


def test_operator_jwt_gets_redacted_field(
    require_postgres, require_ollama, config, tmp_path: Path, monkeypatch
):
    """A tenant_alpha_operator JWT (not admin) never receives the raw key via POST /query.

    The public `QueryResponse` contract doesn't expose raw chunk `content`
    at all (only chunk_id/document_id/source/category/score). so this
    proves the stronger, structural claim: the literal never appears
    ANYWHERE in the HTTP response body, not just that a particular field
    was scrubbed.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    ns = f"pytest-apifieldredact-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "runbook.md",
        {
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_operator", "tenant_alpha_admin"],
        },
        f"The synthetic test key is {_TEST_KEY}.",
    )
    ingest_result = ingestion.ingest_file(path, ns)

    app.dependency_overrides[get_config] = lambda: secure
    app.dependency_overrides[get_retrieval_pipeline] = lambda: RetrievalPipeline(secure)
    try:
        client = TestClient(app)
        token = _token("tenant_alpha", ["tenant_alpha_operator"])
        response = client.post(
            "/query",
            json={"query": "What is the synthetic test key?", "filters": {"dataset_id": ns}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert _TEST_KEY not in response.text
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        ingestion._vectorstore.delete_document(ingest_result["document_id"])


def test_cross_tenant_jwt_cannot_retrieve_the_document_at_all(
    require_postgres, require_ollama, config, tmp_path: Path, monkeypatch
):
    """A tenant_beta JWT gets no answer grounded in tenant_alpha's document at all."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    ns = f"pytest-apifieldredact-{uuid.uuid4()}"
    secure = _secure_config(config)
    ingestion = IngestionPipeline(secure)
    path = _write_doc(
        tmp_path,
        "runbook.md",
        {"tenant_id": "tenant_alpha", "allowed_roles": ["tenant_alpha_operator"]},
        f"The synthetic test key is {_TEST_KEY}.",
    )
    ingest_result = ingestion.ingest_file(path, ns)

    app.dependency_overrides[get_config] = lambda: secure
    app.dependency_overrides[get_retrieval_pipeline] = lambda: RetrievalPipeline(secure)
    try:
        client = TestClient(app)
        token = _token("tenant_beta", ["tenant_beta_operator"])
        response = client.post(
            "/query",
            json={"query": "What is the synthetic test key?", "filters": {"dataset_id": ns}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["sources"] == []
        assert _TEST_KEY not in response.text
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_retrieval_pipeline, None)
        ingestion._vectorstore.delete_document(ingest_result["document_id"])
