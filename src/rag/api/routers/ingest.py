"""`POST /ingest`: upload files and run them through `IngestionPipeline`."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from rag.api.auth import VerifiedIdentity
from rag.api.deps import get_config, get_current_identity, get_ingestion_pipeline
from rag.audit import log_audit_event, pseudonymous_subject
from rag.config import AppConfig
from rag.ingestion.pipeline import IngestionPipeline

router = APIRouter()

# Uploads land here under their original filename (path components stripped,
# see `_safe_upload_path`), so re-uploading the same file keeps the same
# `source` identity (and therefore the same document_id). Checksum
# comparison decides whether it needs re-indexing.
UPLOAD_DIR = Path("data/uploads")

_UPLOAD_CHUNK_BYTES = 1024 * 1024


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
    """
    _enforce_ingest_authorization(identity, config)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    max_upload_bytes = config.security.dos_limits.max_upload_bytes
    results = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Uploaded file is missing a filename")
        dest = _safe_upload_path(upload.filename)
        await _save_upload_bounded(upload, dest, max_upload_bytes)
        result = pipeline.ingest_file(dest, dataset_id)
        results.append(IngestResult(filename=upload.filename, **result))
    return results
