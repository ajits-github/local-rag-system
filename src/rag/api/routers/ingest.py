from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from rag.api.deps import get_ingestion_pipeline
from rag.ingestion.pipeline import IngestionPipeline

router = APIRouter()

# Uploads land here under their original filename, so re-uploading the same
# file keeps the same `source` identity (and therefore the same
# document_id) — checksum comparison decides whether it needs re-indexing.
UPLOAD_DIR = Path("data/uploads")


class IngestResult(BaseModel):
    filename: str
    document_id: str
    chunks_written: int
    changed: bool


@router.post("/ingest", response_model=list[IngestResult])
async def ingest(
    dataset_id: str = Form(..., description="Namespace tag stored on every chunk (e.g. 'techfusion')."),
    files: list[UploadFile] = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> list[IngestResult]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for upload in files:
        dest = UPLOAD_DIR / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        result = pipeline.ingest_file(dest, dataset_id)
        results.append(IngestResult(filename=upload.filename, **result))
    return results
