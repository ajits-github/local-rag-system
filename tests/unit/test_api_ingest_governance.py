"""`POST /ingest`'s caller-governance wiring: real JWTs, mocked `IngestionPipeline`.

Mirrors `test_api_query_auth_boundary.py`'s pattern (real JWT minting/
verification through the actual `get_current_identity` dependency chain, no
real Postgres/Ollama). Focused on `_build_ingest_caller_context` and the
`IngestGovernanceError` -> 403 mapping in `api/routers/ingest.py`, not on
`IngestionPipeline`'s own governance logic (covered by
`test_ingestion_pipeline_governance.py`) or the pure resolver
(`test_ingest_governance.py`).
"""

from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_ingestion_pipeline
from rag.api.main import app
from rag.config import load_config
from rag.ingestion.governance import IngestCallerContext, IngestGovernanceError

_SECRET = "unit-test-only-not-a-real-secret-value"


class _RecordingIngestionPipeline:
    """IngestionPipeline double recording every `ingest_file()` call's `caller` argument."""

    def __init__(self, *, reject: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reject = reject

    def ingest_file(
        self,
        path,
        dataset_id,
        category=None,
        caller: IngestCallerContext | None = None,
        source_override: str | None = None,
    ):
        self.calls.append(
            {
                "path": path,
                "dataset_id": dataset_id,
                "caller": caller,
                "source_override": source_override,
            }
        )
        if self._reject:
            raise IngestGovernanceError("document tenant_id does not match caller's tenant")
        return {"document_id": "doc-1", "chunks_written": 1, "changed": True}


def _auth_config(**overrides):
    """Load config with security.auth enabled and HS256 configured for tests."""
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    auth_config = auth_config.model_copy(update=overrides)
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _no_auth_config():
    """Load config with security.auth explicitly disabled."""
    config = load_config()
    auth_config = config.security.auth.model_copy(update={"enabled": False})
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _token(**claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


@pytest.fixture
def client_with(monkeypatch, tmp_path):
    """Build a TestClient with `get_config`/`get_ingestion_pipeline` overridden.

    Returns a `(client, pipeline)` factory: `client_with(config, reject=...)`
    installs that config and a fresh `_RecordingIngestionPipeline`, cleaning
    up overrides after the test. `UPLOAD_DIR` is redirected to `tmp_path` so
    nothing touches the real `data/uploads/`.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)
    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)

    def _build(config, *, reject: bool = False):
        pipeline = _RecordingIngestionPipeline(reject=reject)
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
        return TestClient(app), pipeline

    yield _build
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_ingestion_pipeline, None)


def _upload(client, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/ingest",
        data={"dataset_id": "test-dataset"},
        files={"files": ("doc.md", b"content", "text/markdown")},
        headers=headers,
    )


def test_normal_caller_context_carries_own_tenant_and_no_privilege(client_with):
    """A caller with no cross_tenant_support_roles role gets is_privileged=False. [Test A].

    `security_admin` is (by default) in `ingest_roles` (so the request
    clears `_enforce_ingest_authorization`) but not in
    `cross_tenant_support_roles`, the role list this fix's privilege
    check reads, so it's a clean stand-in for "an ingest-eligible caller
    who isn't cross-tenant-privileged."
    """
    client, pipeline = client_with(_auth_config())
    token = _token(tenant_id="tenant_alpha", roles=["security_admin"])

    response = _upload(client, token)

    assert response.status_code == 200
    assert len(pipeline.calls) == 1
    caller = pipeline.calls[0]["caller"]
    assert caller.tenant_id == "tenant_alpha"
    assert caller.is_privileged is False


def test_cross_tenant_support_role_marks_caller_privileged(client_with):
    """A caller holding the configured cross_tenant_support_roles role is privileged. [Test F]."""
    client, pipeline = client_with(_auth_config())
    token = _token(tenant_id="techfusion", roles=["techfusion_support"])

    response = _upload(client, token)

    assert response.status_code == 200
    caller = pipeline.calls[0]["caller"]
    assert caller.tenant_id == "techfusion"
    assert caller.is_privileged is True


def test_governance_rejection_maps_to_403_and_stops_the_batch(client_with):
    """An IngestGovernanceError from the pipeline becomes a 403, not a 500. [Test E]."""
    client, pipeline = client_with(_auth_config(), reject=True)
    token = _token(tenant_id="tenant_alpha", roles=["security_admin"])

    response = _upload(client, token)

    assert response.status_code == 403
    assert "tenant" in response.json()["detail"]
    assert len(pipeline.calls) == 1  # the rejected file was attempted, nothing further


def test_auth_disabled_passes_no_caller_context(client_with):
    """`security.auth.enabled=False`: no identity, so no caller context at all. [Test G]."""
    client, pipeline = client_with(_no_auth_config())

    response = _upload(client)

    assert response.status_code == 200
    assert pipeline.calls[0]["caller"] is None


def test_insecure_dev_mode_with_no_token_passes_no_caller_context(client_with):
    """auth.enabled + insecure_dev_mode + no Authorization header -> no caller. [Test G]."""
    client, pipeline = client_with(_auth_config(insecure_dev_mode=True))

    response = _upload(client)  # no token

    assert response.status_code == 200
    assert pipeline.calls[0]["caller"] is None
