"""Scheduled directory scan against a real PostgreSQL and the real IngestionPipeline."""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from pathlib import Path

import pytest

from rag.config import ScanSourceConfig, ScheduledScanConfig
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.scheduler import scan_source


class _ConstantEmbedder:
    """Fixed vector per text; keeps the test free of model downloads."""

    def __init__(self, dimension: int) -> None:
        self._vector = [1.0] + [0.0] * (dimension - 1)

    def embed_query(self, text: str) -> list[float]:
        """Return the constant vector."""
        return self._vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return the constant vector per text."""
        return [self._vector for _ in texts]


@pytest.fixture
def setup(require_postgres, config, tmp_path):
    """Return a pipeline factory, a populated root, a scan config, and a source."""
    dataset_id = f"pytest-sched-{uuid.uuid4()}"
    root = tmp_path / "kb"
    (root / "ops").mkdir(parents=True)
    (root / "a.md").write_text("# A\n\nAlpha retention is 30 days.", encoding="utf-8")
    (root / "ops" / "b.md").write_text("# B\n\nBravo runbook.", encoding="utf-8")
    scan_config = ScheduledScanConfig(
        enabled=True, state_dir=str(tmp_path / "state"), settle_seconds=0, max_delete_fraction=1.0
    )
    source = ScanSourceConfig(path=str(root), dataset_id=dataset_id)

    def make_pipeline():
        return IngestionPipeline(config, embedder=_ConstantEmbedder(config.embedding.dimension))

    first = make_pipeline()
    try:
        yield make_pipeline, first, root, scan_config, source
    finally:
        first.clear_dataset(dataset_id)


def _scan(pipeline, scan_config, source, config):
    return scan_source(pipeline, source, scan_config, config.ingestion.supported_extensions)


def test_lifecycle_new_unchanged_changed_deleted(config, setup):
    """One source through add, no-op, edit, and delete rounds."""
    make_pipeline, pipeline, root, scan_config, source = setup
    store = pipeline.vectorstore

    report = _scan(pipeline, scan_config, source, config)
    assert (report.status, report.new) == ("completed", 2)
    ids_before = {
        s: store.get_or_create_document_id(s, "probe", source.dataset_id)[0]
        for s in store.list_document_sources(source.dataset_id)
    }

    assert _scan(pipeline, scan_config, source, config).unchanged == 2

    (root / "a.md").write_text("# A\n\nAlpha retention is 90 days.", encoding="utf-8")
    report = _scan(pipeline, scan_config, source, config)
    assert (report.changed, report.unchanged) == (1, 1)
    a_source = str(root / "a.md")
    chunks = store.get_chunks_by_source(a_source, source.dataset_id)
    assert any("90 days" in c.content for c in chunks)
    assert not any("30 days" in c.content for c in chunks)
    assert (
        store.get_or_create_document_id(a_source, "probe", source.dataset_id)[0]
        == (ids_before[a_source])
    )

    (root / "ops" / "b.md").unlink()
    report = _scan(pipeline, scan_config, source, config)
    assert report.deleted == 1
    assert store.list_document_sources(source.dataset_id) == [a_source]


def test_restart_with_a_fresh_pipeline_and_no_manifest_is_a_no_op(config, setup):
    """A process restart (new pipeline, lost state dir) neither deletes nor re-ingests."""
    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    shutil.rmtree(scan_config.state_dir)
    report = _scan(make_pipeline(), scan_config, source, config)
    assert (report.status, report.unchanged, report.new, report.deleted) == ("completed", 2, 0, 0)


def test_unavailable_root_keeps_every_document(config, setup):
    """A vanished directory is reported, and the dataset is left untouched."""
    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    shutil.rmtree(root)
    report = _scan(pipeline, scan_config, source, config)
    assert report.status == "root_unavailable"
    assert len(pipeline.vectorstore.list_document_sources(source.dataset_id)) == 2


def test_emptied_root_keeps_every_document(config, setup):
    """An empty directory (unmounted share) withholds deletion even at max_delete_fraction=1."""
    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    (root / "a.md").unlink()
    (root / "ops" / "b.md").unlink()
    report = _scan(pipeline, scan_config, source, config)
    assert report.deletion_blocked_reason == "no_files_found"
    assert len(pipeline.vectorstore.list_document_sources(source.dataset_id)) == 2


def test_concurrent_triggers_from_two_pipelines_do_not_double_ingest(config, setup):
    """Two pipelines (separate pools) racing on one source: one scans, the other is locked."""
    make_pipeline, pipeline, root, scan_config, source = setup
    for i in range(15):
        (root / f"extra{i}.md").write_text(f"# E{i}\n\nExtra {i}.", encoding="utf-8")
    other = make_pipeline()
    reports = []
    threads = [
        threading.Thread(target=lambda p=p: reports.append(_scan(p, scan_config, source, config)))
        for p in (pipeline, other)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    statuses = sorted(r.status for r in reports)
    assert statuses in (["completed", "locked"], ["completed", "completed"])
    total_new = sum(r.new for r in reports)
    assert total_new == 17
    assert len(pipeline.vectorstore.list_document_sources(source.dataset_id)) == 17
    assert not list(Path(scan_config.state_dir).glob("*.lock"))


class _FailOnceEmbedder(_ConstantEmbedder):
    """Raises on the first batch containing `marker`, then behaves normally."""

    def __init__(self, dimension: int, marker: str) -> None:
        super().__init__(dimension)
        self._marker = marker
        self.failed = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Fail once for the marked document, as an interrupted ingestion would."""
        if not self.failed and any(self._marker in t for t in texts):
            self.failed = True
            raise RuntimeError("embedder interrupted")
        return super().embed_documents(texts)


def _chunk_text(pipeline, source: str, dataset_id: str) -> str:
    return " ".join(
        c.content for c in pipeline.vectorstore.get_chunks_by_source(source, dataset_id)
    )


def test_atomic_replacement_with_identical_size_and_mtime_is_detected(config, setup):
    """Write-then-rename replacement is re-ingested even when size and mtime match."""
    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    target = root / "a.md"
    before = target.stat()
    staged = root / "a.md.tmp"
    staged.write_text("# A\n\nAlpha retention is 60 days.", encoding="utf-8")
    assert staged.stat().st_size == before.st_size
    os.utime(staged, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(staged, target)
    report = _scan(pipeline, scan_config, source, config)
    assert report.changed == 1
    assert "60 days" in _chunk_text(pipeline, str(target), source.dataset_id)


def test_temporary_and_lock_files_never_become_documents(config, setup):
    """Editor swap files, Office lock files, and partial downloads are skipped."""
    make_pipeline, pipeline, root, scan_config, source = setup
    for name in ("~$a.md", ".a.md.swp", "c.md.part", "d.md~"):
        (root / name).write_text("temporary", encoding="utf-8")
    report = _scan(pipeline, scan_config, source, config)
    stored = {Path(s).name for s in pipeline.vectorstore.list_document_sources(source.dataset_id)}
    assert stored == {"a.md", "b.md"} and report.ignored == 4


def test_rename_replaces_the_document_and_counts_a_probable_rename(config, setup):
    """A rename is delete plus add: the old path disappears, the new one is searchable."""
    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    (root / "a.md").rename(root / "a-renamed.md")
    report = _scan(pipeline, scan_config, source, config)
    assert (report.new, report.deleted, report.probable_renames) == (1, 1, 1)
    stored = {Path(s).name for s in pipeline.vectorstore.list_document_sources(source.dataset_id)}
    assert stored == {"a-renamed.md", "b.md"}


def test_interrupted_ingestion_is_retried_on_the_next_round(config, setup):
    """An embedder failure leaves a pending checksum, so the next round re-ingests the file."""
    make_pipeline, pipeline, root, scan_config, source = setup
    flaky = IngestionPipeline(
        config, embedder=_FailOnceEmbedder(config.embedding.dimension, "Bravo")
    )
    report = _scan(flaky, scan_config, source, config)
    assert report.status == "partial" and report.failed == 1 and report.new == 1
    b_source = str(root / "ops" / "b.md")
    assert pipeline.vectorstore.get_chunks_by_source(b_source, source.dataset_id) == []
    report = _scan(flaky, scan_config, source, config)
    assert report.status == "completed" and report.changed == 1 and report.unchanged == 1
    assert "Bravo" in _chunk_text(pipeline, b_source, source.dataset_id)


def test_partial_directory_failure_ingests_the_rest_but_deletes_nothing(config, setup, monkeypatch):
    """An unreadable subdirectory withholds deletion while readable files still ingest."""
    from rag.ingestion import pipeline as pipeline_module

    make_pipeline, pipeline, root, scan_config, source = setup
    _scan(pipeline, scan_config, source, config)
    (root / "ops" / "b.md").unlink()
    (root / "new.md").write_text("# N\n\nNew file.", encoding="utf-8")
    real_walk = os.walk

    def walk_with_error(top, onerror=None, **kwargs):
        if onerror is not None:
            onerror(PermissionError(13, "denied", str(Path(top) / "restricted")))
        yield from real_walk(top, onerror=onerror, **kwargs)

    monkeypatch.setattr(pipeline_module.os, "walk", walk_with_error)
    report = _scan(pipeline, scan_config, source, config)
    assert report.new == 1 and report.deleted == 0
    assert report.deletion_blocked_reason == "enumeration_incomplete"
    assert str(root / "ops" / "b.md") in pipeline.vectorstore.list_document_sources(
        source.dataset_id
    )


def test_deletion_threshold_holds_until_raised(config, setup):
    """Removing most of a source is withheld at the default threshold."""
    make_pipeline, pipeline, root, scan_config, source = setup
    for i in range(4):
        (root / f"x{i}.md").write_text(f"# X{i}\n\nExtra {i}.", encoding="utf-8")
    _scan(pipeline, scan_config, source, config)
    for i in range(4):
        (root / f"x{i}.md").unlink()
    (root / "a.md").unlink()
    strict = scan_config.model_copy(update={"max_delete_fraction": 0.5})
    report = _scan(pipeline, strict, source, config)
    assert report.deletion_blocked_reason == "delete_fraction_exceeded" and report.deleted == 0
    report = _scan(pipeline, scan_config, source, config)
    assert report.deleted == 5
