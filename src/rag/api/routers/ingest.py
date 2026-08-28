"""`POST /ingest`: upload files and run them through `IngestionPipeline`."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_ingestion_pipeline
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig
from rag.ingestion.governance import IngestCallerContext, IngestGovernanceError
from rag.ingestion.pipeline import IngestionPipeline

router = APIRouter()

# Uploads land here under their original filename (path components stripped,
# see `_safe_upload_path`), so re-uploading the same file keeps the same
# `source` identity (and therefore the same document_id). Checksum
# comparison decides whether it needs re-indexing. The upload itself is
# never streamed directly to this final path -- see `_ingest_upload_atomically`.
UPLOAD_DIR = Path("data/uploads")

_UPLOAD_CHUNK_BYTES = 1024 * 1024
_TMP_UPLOAD_PREFIX = ".upload-tmp-"


class IngestResult(BaseModel):
    """Per-file outcome of an ingest request."""

    filename: str
    document_id: str
    chunks_written: int
    changed: bool


def _safe_upload_path(filename: str) -> Path:
    """Return `UPLOAD_DIR / <basename of filename>`, stripping any path components.

    Parameters
    ----------
    filename : str
        The client-supplied original filename.

    Returns
    -------
    Path
        A destination path guaranteed to stay inside `UPLOAD_DIR`, even if
        `filename` contains `../` or an absolute path.
    """
    return UPLOAD_DIR / Path(filename).name


async def _save_upload_bounded(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    """Stream `upload` to `dest`, rejecting it once it exceeds `max_bytes`.

    Parameters
    ----------
    upload : UploadFile
        The incoming file upload.
    dest : Path
        Destination path to write to.
    max_bytes : int
        Maximum allowed upload size, in bytes.

    Raises
    ------
    HTTPException
        413, once the streamed byte count exceeds `max_bytes`. The
        partial file is removed rather than left on disk.
    """
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                log_audit_event("oversized_request_rejected", field="upload", limit=max_bytes)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds maximum size of {max_bytes} bytes",
                )
            f.write(chunk)


def _enforce_ingest_authorization(identity: VerifiedIdentity | None, config: AppConfig) -> None:
    """Reject an ingest request whose verified roles aren't allow-listed.

    A no-op unless `security.auth.enabled=True` **and** a verified
    identity was actually supplied. Ingestion stays wide-open by default
    when auth is disabled or when `insecure_dev_mode` let an
    unauthenticated caller through.

    Parameters
    ----------
    identity : VerifiedIdentity | None
        The verified caller identity, or `None`.
    config : AppConfig
        Application configuration.

    Raises
    ------
    HTTPException
        403, when `identity.roles` doesn't intersect `security.auth.ingest_roles`.
    """
    if not config.security.auth.enabled or identity is None:
        return
    allowed = set(config.security.auth.ingest_roles)
    if not allowed.intersection(identity.roles):
        log_audit_event(
            "authorization_denied",
            subject=pseudonymous_subject(identity.subject),
            action="ingest",
            tenant_id=identity.tenant_id,
        )
        raise HTTPException(
            status_code=403, detail="Caller's roles are not authorized to ingest documents"
        )


def _temp_upload_path(dest: Path) -> Path:
    """Return a unique temp path in the same directory as `dest`.

    Same directory (and therefore filesystem/volume) as `dest`, so the
    final `os.replace` in `_ingest_upload_atomically` is a true atomic
    rename rather than a cross-filesystem copy+delete.

    Parameters
    ----------
    dest : Path
        The final, stable destination path.

    Returns
    -------
    Path
    """
    return dest.with_name(f"{_TMP_UPLOAD_PREFIX}{uuid.uuid4().hex}{dest.suffix}")


async def _ingest_upload_atomically(
    upload: UploadFile,
    dest: Path,
    dataset_id: str,
    pipeline: IngestionPipeline,
    caller: IngestCallerContext | None,
    max_upload_bytes: int,
) -> dict[str, Any]:
    """Stream, ingest, and install an upload without ever writing directly to `dest`.

    The upload lands at a unique temporary path in `UPLOAD_DIR` first.
    Size-bounding, loader parsing, governance resolution, and the
    ingestion DB write all run against that temp file (with the
    persisted document `source` overridden back to `dest`'s stable path,
    via `IngestionPipeline.ingest_file`'s `source_override`, so
    re-uploading the same filename still resolves to the same
    `document_id`). `dest` is only touched, via an atomic `os.replace`,
    once ingestion has fully succeeded. Any failure along the way
    (oversized upload, unsupported extension, parse failure, cross-tenant
    governance rejection) removes only the temp file and leaves an
    existing `dest` byte-for-byte untouched.

    Parameters
    ----------
    upload : UploadFile
        The incoming file upload.
    dest : Path
        Final, stable destination path -- the document's `source` identity.
    dataset_id : str
        Namespace tag stored on every chunk.
    pipeline : IngestionPipeline
        Ingestion pipeline to run the temp file through.
    caller : IngestCallerContext | None
        Governance context; see `IngestionPipeline.ingest_file`.
    max_upload_bytes : int
        Maximum allowed upload size, in bytes.

    Returns
    -------
    dict[str, Any]
        `IngestionPipeline.ingest_file`'s result dict.

    Raises
    ------
    HTTPException
        413, if the upload exceeds `max_upload_bytes`.
    IngestGovernanceError
        Propagated from `ingest_file` for the caller to map to a 403.
    """
    tmp_path = _temp_upload_path(dest)
    try:
        await _save_upload_bounded(upload, tmp_path, max_upload_bytes)
        result = pipeline.ingest_file(
            tmp_path, dataset_id, caller=caller, source_override=str(dest)
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, dest)
    return result


def _build_ingest_caller_context(
    identity: VerifiedIdentity | None, config: AppConfig
) -> IngestCallerContext | None:
    """Build the governance context `IngestionPipeline.ingest_file` resolves `tenant_id` from.

    `None` whenever there's no verified identity at all (JWT auth disabled,
    or `insecure_dev_mode` let an unauthenticated request through) --
    ingestion then behaves exactly as it did before this fix, matching
    every other identity-less ingestion path (the CLI, `make ingest`).

    Parameters
    ----------
    identity : VerifiedIdentity | None
        The verified caller identity, or `None`.
    config : AppConfig
        Application configuration; reads
        `security.authorization.cross_tenant_support_roles`, the same role
        list retrieval-time cross-tenant access already uses.

    Returns
    -------
    IngestCallerContext | None
    """
    if identity is None:
        return None
    privileged = bool(
        set(identity.roles) & set(config.security.authorization.cross_tenant_support_roles)
    )
    return IngestCallerContext(tenant_id=identity.tenant_id, is_privileged=privileged)


@router.post("/ingest", response_model=list[IngestResult])
async def ingest(
    dataset_id: str = Form(
        ..., description="Namespace tag stored on every chunk (e.g. 'techfusion')."
    ),
    files: list[UploadFile] = File(...),
    identity: VerifiedIdentity | None = Depends(get_current_identity),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    config: AppConfig = Depends(get_config),
) -> list[IngestResult]:
    """Save each upload to `UPLOAD_DIR` and ingest it under `dataset_id`.

    Parameters
    ----------
    dataset_id : str
        Namespace tag stored on every chunk.
    files : list[UploadFile]
        Uploaded files to ingest.
    identity : VerifiedIdentity | None
        The verified caller identity (see `get_current_identity`), or
        `None` when JWT auth is disabled.
    pipeline : IngestionPipeline
        Injected ingestion pipeline singleton.
    config : AppConfig
        Application configuration.

    Returns
    -------
    list[IngestResult]
        One result per uploaded file, in the same order.

    Raises
    ------
    HTTPException
        403, when an authenticated caller's document (via front matter, if
        any) specifies a `tenant_id` different from their own verified
        tenant and they hold no cross-tenant support role (see
        `rag.ingestion.governance`).
    """
    _enforce_ingest_authorization(identity, config)
    caller = _build_ingest_caller_context(identity, config)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    max_upload_bytes = config.security.dos_limits.max_upload_bytes
    results = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Uploaded file is missing a filename")
        dest = _safe_upload_path(upload.filename)
        try:
            result = await _ingest_upload_atomically(
                upload, dest, dataset_id, pipeline, caller, max_upload_bytes
            )
        except IngestGovernanceError as exc:
            assert identity is not None  # caller is only non-None when identity is
            log_audit_event(
                "cross_tenant_attempt",
                subject=pseudonymous_subject(identity.subject),
                action="ingest",
                verified_tenant_id=identity.tenant_id,
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        results.append(IngestResult(filename=upload.filename, **result))
    return results
