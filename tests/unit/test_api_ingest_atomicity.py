"""`POST /ingest`'s upload lifecycle: never write directly to the final path.

Focused on the M3/C1-adjacent atomicity fix in `api/routers/ingest.py`
(`_ingest_upload_atomically`/`_temp_upload_path`): an upload always lands at
a unique temp file in `UPLOAD_DIR` first, and the final destination is only
touched via an atomic `os.replace`, once ingestion (loader parsing,
governance resolution, DB write) has fully succeeded. A rejected or failed
upload must never truncate, delete, or partially overwrite a
previously-accepted file under the same name.

Uses a fake `IngestionPipeline` (no real Postgres/Ollama), so these tests
prove the router's own file-lifecycle behavior, not `IngestionPipeline`'s
internal governance logic (covered by `test_ingestion_pipeline_governance.py`)
or `resolve_ingest_tenant_id` itself (covered by `test_ingest_governance.py`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_ingestion_pipeline
from rag.api.main import app
from rag.config import load_config
from rag.ingestion.governance import IngestGovernanceError

_SECRET = "unit-test-only-not-a-real-secret-value"


class _FakeAtomicPipeline:
    """IngestionPipeline double that records each `ingest_file()` call's arguments.

    Also snapshots the temp `path` it was called with (existence + bytes)
    at call time, so a test can prove the router streamed the real upload
    content to a temp file *before* calling `ingest_file`, rather than
    ever handing it the final destination path directly.
    """

    def __init__(self, *, reject: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reject = reject

    def ingest_file(
        self, path, dataset_id, category=None, caller=None, source_override=None
    ) -> dict[str, Any]:
        path = Path(path)
        self.calls.append(
            {
                "path": path,
                "dataset_id": dataset_id,
                "caller": caller,
                "source_override": source_override,
                "path_existed_at_call": path.exists(),
                "path_bytes_at_call": path.read_bytes() if path.exists() else None,
            }
        )
        if self._reject:
            raise IngestGovernanceError("document tenant_id does not match caller's tenant")
        return {"document_id": "doc-1", "chunks_written": 3, "changed": True}


def _auth_config(**overrides):
    """Load config with security.auth enabled and HS256 configured for tests."""
    config = load_config()
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    auth_config = auth_config.model_copy(update=overrides)
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _no_auth_config():
    """Load config with security.auth explicitly disabled; caller=None on every call."""
    config = load_config()
    auth_config = config.security.auth.model_copy(update={"enabled": False})
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _token(**claim_overrides):
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["security_admin"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Redirect `UPLOAD_DIR` to an isolated `tmp_path`, real filesystem I/O."""
    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def client_with(monkeypatch, upload_dir):
    """Build a `TestClient` with `get_config`/`get_ingestion_pipeline` overridden.

    Returns a `(client, pipeline)` factory: `client_with(config, reject=...)`.
    """
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)

    def _build(config, *, reject: bool = False):
        pipeline = _FakeAtomicPipeline(reject=reject)
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
        return TestClient(app), pipeline

    yield _build
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_ingestion_pipeline, None)


def _no_leftover_tmp_files(upload_dir: Path) -> bool:
    return not any(upload_dir.glob(".upload-tmp-*"))


def _upload(client, filename: str, content: bytes, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/ingest",
        data={"dataset_id": "test-dataset"},
        files={"files": (filename, content, "text/markdown")},
        headers=headers,
    )


def test_rejected_cross_tenant_upload_leaves_no_new_final_file(client_with, upload_dir):
    """A governance-rejected upload of a brand-new filename creates no final file. [Test A]."""
    client, pipeline = client_with(_auth_config(), reject=True)
    token = _token()

    response = _upload(client, "new-doc.md", b"attempted cross-tenant content", token)

    assert response.status_code == 403
    assert not (upload_dir / "new-doc.md").exists()
    assert _no_leftover_tmp_files(upload_dir)
    assert len(pipeline.calls) == 1


def test_rejected_cross_tenant_same_filename_preserves_previous_file(client_with, upload_dir):
    """A governance-rejected re-upload never touches the existing accepted file. [Test B]."""
    existing = upload_dir / "existing.md"
    existing.write_bytes(b"ORIGINAL ACCEPTED CONTENT")
    client, pipeline = client_with(_auth_config(), reject=True)
    token = _token()

    response = _upload(client, "existing.md", b"NEW REJECTED CONTENT", token)

    assert response.status_code == 403
    assert existing.read_bytes() == b"ORIGINAL ACCEPTED CONTENT"
    assert _no_leftover_tmp_files(upload_dir)
    assert len(pipeline.calls) == 1


def test_oversized_same_filename_upload_preserves_previous_file(client_with, upload_dir):
    """An oversized re-upload never touches the existing accepted file. [Test C]."""
    existing = upload_dir / "existing.md"
    existing.write_bytes(b"ORIGINAL ACCEPTED CONTENT")
    small_limit_config = _no_auth_config().model_copy(
        update={
            "security": _no_auth_config().security.model_copy(
                update={
                    "dos_limits": _no_auth_config().security.dos_limits.model_copy(
                        update={"max_upload_bytes": 10}
                    )
                }
            )
        }
    )
    client, pipeline = client_with(small_limit_config)

    response = _upload(client, "existing.md", b"x" * 1000)

    assert response.status_code == 413
    assert existing.read_bytes() == b"ORIGINAL ACCEPTED CONTENT"
    assert _no_leftover_tmp_files(upload_dir)
    assert pipeline.calls == []  # oversized rejection never reaches ingest_file


def test_successful_same_filename_upload_atomically_replaces_old_file(client_with, upload_dir):
    """A successful re-upload atomically replaces the old file's content. [Test D]."""
    existing = upload_dir / "doc.md"
    existing.write_bytes(b"OLD CONTENT")
    client, pipeline = client_with(_no_auth_config())

    response = _upload(client, "doc.md", b"NEW CONTENT")

    assert response.status_code == 200
    assert existing.read_bytes() == b"NEW CONTENT"
    assert _no_leftover_tmp_files(upload_dir)
    assert len(pipeline.calls) == 1


def test_rejected_governance_upload_never_reaches_a_second_attempt(client_with, upload_dir):
    """A governance rejection stops the batch; the fake records exactly one attempt. [Test E].

    The deeper claim, that `IngestGovernanceError` itself is raised before
    any DB write, is proven at the `IngestionPipeline` level by
    `test_ingestion_pipeline_governance.py::
    test_normal_caller_cannot_upload_front_matter_for_a_different_tenant`,
    which asserts `vectorstore.written_chunks == []`. This test proves the
    router-level consequence: no final file, no leftover temp file, and no
    further ingestion work performed for that upload.
    """
    client, pipeline = client_with(_auth_config(), reject=True)
    token = _token()

    response = _upload(client, "rejected.md", b"content", token)

    assert response.status_code == 403
    assert not (upload_dir / "rejected.md").exists()
    assert len(pipeline.calls) == 1


def test_successful_upload_persists_final_source_path_not_a_temp_filename(client_with, upload_dir):
    """`ingest_file` is called with the temp path but `source_override` is the stable path.

    Proves the router never lets a temp filename leak into the document's
    persisted `source` identity. [Test F]
    """
    client, pipeline = client_with(_no_auth_config())

    response = _upload(client, "report.pdf", b"%PDF-1.4 fake pdf bytes")

    assert response.status_code == 200
    call = pipeline.calls[0]
    assert call["source_override"] == str(upload_dir / "report.pdf")
    assert ".upload-tmp-" in str(call["path"])
    assert call["path"] != upload_dir / "report.pdf"
    # The temp file ingest_file was actually called against held the real upload bytes.
    assert call["path_existed_at_call"] is True
    assert call["path_bytes_at_call"] == b"%PDF-1.4 fake pdf bytes"
    assert _no_leftover_tmp_files(upload_dir)
