"""Fault injection for SEC-007: a crash between the DB commit and the final file install.

`_ingest_upload_atomically` commits the document to Postgres, then installs the
staged file with `os.replace`. A crash in between leaves the database one step
ahead of the filesystem. These tests simulate that crash at the install step and
prove the resulting state is detectable and that retrying the same upload heals
it. CI-safe: synthetic Markdown in a disposable dataset; no Ollama or Neo4j.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import uuid
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from rag.api.routers import ingest as ingest_router
from rag.ingestion.pipeline import IngestionPipeline

_MAX_BYTES = 10_000_000


class _SimulatedCrash(BaseException):
    """Stands in for the process dying; nothing after it runs."""


def _upload(data: bytes, name: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


def _run(upload: UploadFile, dest: Path, ns: str, pipeline: IngestionPipeline):
    return asyncio.run(
        ingest_router._ingest_upload_atomically(upload, dest, ns, pipeline, None, _MAX_BYTES)
    )


def _crash_on_final_install(monkeypatch, dest: Path) -> None:
    real_replace = os.replace

    def replace(src, dst, *args, **kwargs):
        if Path(dst) == dest:
            raise _SimulatedCrash()
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ingest_router.os, "replace", replace)


_PREVIOUS = b"# Report\n\nThe retention period is 7 days.\n"


@pytest.mark.parametrize("previous_version", [None, _PREVIOUS], ids=["first-upload", "replacement"])
def test_crash_after_db_commit_is_detectable_and_healed_by_retrying_the_upload(
    require_postgres, config, tmp_path: Path, monkeypatch, previous_version
):
    """DB ahead of the file after the crash; a retry installs the file and changes nothing else."""
    ns = f"pytest-crash-window-{uuid.uuid4()}"
    pipeline = IngestionPipeline(config)
    store = pipeline._vectorstore
    dest = tmp_path / "report.md"
    new_bytes = b"# Report\n\nThe retention period is 30 days.\n"
    if previous_version is not None:
        _run(_upload(previous_version, dest.name), dest, ns, pipeline)
        assert dest.read_bytes() == previous_version

    _crash_on_final_install(monkeypatch, dest)
    with pytest.raises(_SimulatedCrash):
        _run(_upload(new_bytes, dest.name), dest, ns, pipeline)
    try:
        # The window is real: Postgres already holds the new version...
        checksum = store.get_document_checksums(ns)[str(dest)]
        assert checksum == hashlib.sha256(new_bytes).hexdigest()
        # ...while the installed file is the previous version, or absent.
        if previous_version is None:
            assert not dest.exists()
        else:
            assert dest.read_bytes() == previous_version
        # Detectable: the stored checksum disagrees with the file on disk, and
        # the orphaned staging directory is still there.
        assert not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != checksum
        assert list(tmp_path.glob(".upload-tmp-*"))
        # Retrieval is unaffected: chunks live in Postgres, so the new content is served.
        chunks = store.get_chunks_by_source(str(dest), ns)
        assert any("30 days" in c.content for c in chunks)
        assert not any("7 days" in c.content for c in chunks)

        # Healed by retrying the same upload: unchanged in Postgres, file installed.
        monkeypatch.undo()
        result = _run(_upload(new_bytes, dest.name), dest, ns, pipeline)
        assert result["changed"] is False
        assert dest.read_bytes() == new_bytes
    finally:
        pipeline.clear_dataset(ns)
