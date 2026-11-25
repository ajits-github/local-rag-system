"""Scheduled directory scan: change detection, deletion safeguards, locking, and restarts."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import pytest

from rag.config import ScanSourceConfig, ScheduledScanConfig
from rag.ingestion import scheduler
from rag.ingestion.scheduler import scan_source

_EXTS = [".md", ".txt"]


class FakePipeline:
    """In-memory stand-in for `IngestionPipeline`: `db` maps source -> checksum."""

    def __init__(self, *, digest_mode: bool = True, fail_on: tuple[str, ...] = ()) -> None:
        self.db: dict[str, str] = {}
        self.ingest_calls: list[dict] = []
        self.delete_calls: list[set[str]] = []
        self.checksum_is_file_digest = digest_mode
        self._fail_on = fail_on
        self.vectorstore = self
        self.on_ingest = None

    def get_document_checksums(self, dataset_id: str) -> dict[str, str]:
        """Return a copy of the in-memory checksum table."""
        return dict(self.db)

    def ingest_file(self, path, dataset_id, category=None, caller=None, source_override=None):
        """Record the call and store the file's digest, failing for configured names."""
        source = source_override or str(path)
        self.ingest_calls.append(
            {"source": source, "category": category, "caller": caller, "override": source_override}
        )
        if self.on_ingest is not None:
            self.on_ingest()
        if Path(path).name in self._fail_on:
            raise ValueError("unparsable")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        changed = self.db.get(source) != digest
        self.db[source] = digest
        return {"document_id": source, "chunks_written": 1, "changed": changed}

    def delete_sources(self, dataset_id: str, sources: set[str]) -> int:
        """Remove `sources` from the table."""
        self.delete_calls.append(set(sources))
        for source in sources:
            self.db.pop(source, None)
        return len(sources)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Create a knowledge-base directory with two documents."""
    kb = tmp_path / "kb"
    (kb / "policies").mkdir(parents=True)
    (kb / "a.md").write_text("alpha", encoding="utf-8")
    (kb / "policies" / "b.md").write_text("bravo", encoding="utf-8")
    return kb


def _config(tmp_path: Path, **overrides) -> ScheduledScanConfig:
    values = {
        "enabled": True,
        "state_dir": str(tmp_path / "state"),
        "settle_seconds": 0,
        "max_delete_fraction": 0.5,
    }
    values.update(overrides)
    return ScheduledScanConfig(**values)


def _scan(pipeline, root, tmp_path, *, source=None, **overrides):
    source = source or ScanSourceConfig(path=str(root), dataset_id="ds")
    return scan_source(pipeline, source, _config(tmp_path, **overrides), _EXTS)


def test_first_round_ingests_new_files_with_folder_category(root, tmp_path):
    """New files are ingested once, with the folder-derived category."""
    pipeline = FakePipeline()
    report = _scan(pipeline, root, tmp_path)
    assert report.status == "completed"
    assert (report.new, report.changed, report.unchanged, report.deleted) == (2, 0, 0, 0)
    categories = {Path(c["source"]).name: c["category"] for c in pipeline.ingest_calls}
    assert categories == {"a.md": None, "b.md": "policies"}


def test_zero_settle_window_ingests_a_file_whose_mtime_is_ahead_of_the_clock(root, tmp_path):
    """With `settle_seconds=0`, an mtime slightly ahead of `clock()` does not defer the file."""
    pipeline = FakePipeline()
    newest = max(p.stat().st_mtime for p in root.rglob("*.md"))
    source = ScanSourceConfig(path=str(root), dataset_id="ds")
    report = scan_source(
        pipeline, source, _config(tmp_path), _EXTS, clock=lambda: newest - 0.000001
    )
    assert (report.new, report.deferred) == (2, 0)


def test_unchanged_files_are_skipped_without_loading(root, tmp_path):
    """Second round: stat + manifest + stored checksum agree, so nothing is ingested."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    pipeline.ingest_calls.clear()
    report = _scan(pipeline, root, tmp_path)
    assert report.unchanged == 2 and report.status == "completed"
    assert pipeline.ingest_calls == []


def test_changed_file_is_reingested(root, tmp_path):
    """An edited file is classified as changed and re-ingested under the same source."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "a.md").write_text("alpha v2", encoding="utf-8")
    report = _scan(pipeline, root, tmp_path)
    assert (report.changed, report.unchanged) == (1, 1)
    assert [Path(c["source"]).name for c in pipeline.ingest_calls[-1:]] == ["a.md"]


def test_touched_but_identical_file_is_unchanged(root, tmp_path):
    """A new mtime with identical bytes is rehashed, not re-ingested."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    pipeline.ingest_calls.clear()
    earlier = time.time() - 3600
    os.utime(root / "a.md", (earlier, earlier))
    report = _scan(pipeline, root, tmp_path)
    assert report.unchanged == 2 and pipeline.ingest_calls == []


def test_deleted_file_is_removed_after_a_complete_scan(root, tmp_path):
    """A file gone from a fully enumerated directory is deleted."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "policies" / "b.md").unlink()
    report = _scan(pipeline, root, tmp_path)
    assert report.deleted == 1 and report.status == "completed"
    assert all(not s.endswith("b.md") for s in pipeline.db)


def test_missing_root_never_deletes(root, tmp_path):
    """An unavailable directory is not evidence that its files were deleted."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "a.md").unlink()
    (root / "policies" / "b.md").unlink()
    (root / "policies").rmdir()
    root.rmdir()
    report = _scan(pipeline, root, tmp_path)
    assert report.status == "root_unavailable"
    assert pipeline.delete_calls == [] and len(pipeline.db) == 2


def test_empty_root_never_deletes(root, tmp_path):
    """An empty directory (e.g. an unmounted mount point) withholds deletion."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "a.md").unlink()
    (root / "policies" / "b.md").unlink()
    report = _scan(pipeline, root, tmp_path, max_delete_fraction=1.0)
    assert report.status == "partial"
    assert report.deletion_blocked_reason == "no_files_found"
    assert pipeline.delete_calls == [] and len(pipeline.db) == 2


def test_enumeration_error_never_deletes(root, tmp_path, monkeypatch):
    """An unreadable subdirectory makes the enumeration incomplete, so nothing is deleted."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "policies" / "b.md").unlink()
    real_walk = os.walk

    def walk_with_error(top, onerror=None, **kwargs):
        if onerror is not None:
            onerror(PermissionError(13, "denied", str(Path(top) / "restricted")))
        yield from real_walk(top, onerror=onerror, **kwargs)

    monkeypatch.setattr(scheduler.os, "walk", walk_with_error)
    report = _scan(pipeline, root, tmp_path, max_delete_fraction=1.0)
    assert report.deletion_blocked_reason == "enumeration_incomplete"
    assert report.status == "partial" and pipeline.delete_calls == []
    assert any("PermissionError" in e for e in report.errors)


def test_mass_deletion_above_the_threshold_is_refused(tmp_path):
    """Losing most files at once is withheld until an operator raises the threshold."""
    kb = tmp_path / "kb"
    kb.mkdir()
    for i in range(4):
        (kb / f"d{i}.md").write_text(str(i), encoding="utf-8")
    pipeline = FakePipeline()
    _scan(pipeline, kb, tmp_path)
    for i in range(3):
        (kb / f"d{i}.md").unlink()
    report = _scan(pipeline, kb, tmp_path)
    assert report.deletion_blocked_reason == "delete_fraction_exceeded"
    assert report.deletion_candidates == 3 and pipeline.delete_calls == []
    report = _scan(pipeline, kb, tmp_path, max_delete_fraction=1.0)
    assert report.deleted == 3


def test_failing_file_is_reported_kept_and_does_not_block_other_work(root, tmp_path):
    """One unparsable file: others ingest, it is not deleted, a truly absent file still is."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "c.md").write_text("charlie", encoding="utf-8")
    (root / "a.md").write_text("alpha v2", encoding="utf-8")
    (root / "policies" / "b.md").unlink()
    pipeline._fail_on = ("a.md",)
    report = _scan(pipeline, root, tmp_path, max_delete_fraction=1.0)
    assert report.status == "partial" and report.failed == 1
    assert report.new == 1 and report.deleted == 1
    assert any(s.endswith("a.md") for s in pipeline.db)
    assert any("ValueError" in e for e in report.errors)


def test_settling_file_is_deferred_and_never_deleted(root, tmp_path):
    """A file still being written is skipped this round and kept."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "a.md").write_text("half-written", encoding="utf-8")
    report = _scan(pipeline, root, tmp_path, settle_seconds=3600, max_delete_fraction=1.0)
    assert report.deferred == 2 and report.deleted == 0
    assert pipeline.delete_calls == []


def test_hidden_temporary_and_unsupported_files_are_ignored(root, tmp_path):
    """Editor/Office lock files and hidden folders never become documents."""
    (root / "~$draft.md").write_text("lock", encoding="utf-8")
    (root / "notes.md.swp").write_text("swap", encoding="utf-8")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.md").write_text("hidden", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / "UPPER.MD").write_text("upper", encoding="utf-8")
    pipeline = FakePipeline()
    report = _scan(pipeline, root, tmp_path)
    names = sorted(Path(c["source"]).name for c in pipeline.ingest_calls)
    assert names == ["UPPER.MD", "a.md", "b.md"]
    assert report.ignored == 2


def test_held_lock_skips_the_round(root, tmp_path):
    """A fresh lock file from another process means this round does nothing."""
    pipeline = FakePipeline()
    source = ScanSourceConfig(path=str(root), dataset_id="ds")
    state = tmp_path / "state"
    state.mkdir()
    lock = state / f"{scheduler._source_key(source)}.lock"
    lock.write_text("{}", encoding="utf-8")
    report = _scan(pipeline, root, tmp_path, source=source)
    assert report.status == "locked" and pipeline.ingest_calls == []
    assert lock.exists()


def test_stale_lock_is_taken_over(root, tmp_path):
    """A lock older than `lock_stale_seconds` is assumed dead and replaced."""
    pipeline = FakePipeline()
    source = ScanSourceConfig(path=str(root), dataset_id="ds")
    state = tmp_path / "state"
    state.mkdir()
    lock = state / f"{scheduler._source_key(source)}.lock"
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    report = _scan(pipeline, root, tmp_path, source=source, lock_stale_seconds=60)
    assert report.status == "completed" and not lock.exists()


def test_concurrent_triggers_run_the_scan_once(root, tmp_path):
    """Two overlapping triggers in one process: one scans, the other reports `locked`."""
    pipeline = FakePipeline()
    entered, release = threading.Event(), threading.Event()

    def block_first_ingest():
        if not entered.is_set():
            entered.set()
            release.wait(5)

    pipeline.on_ingest = block_first_ingest
    results = []
    first = threading.Thread(target=lambda: results.append(_scan(pipeline, root, tmp_path)))
    first.start()
    assert entered.wait(5)
    second = _scan(pipeline, root, tmp_path)
    release.set()
    first.join(5)
    assert second.status == "locked"
    assert results[0].status == "completed" and results[0].new == 2


def test_restart_without_manifest_causes_no_deletion_or_reingestion(root, tmp_path):
    """Losing the manifest only disables the fast path; the database decides change."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    for manifest in (tmp_path / "state").glob("*.json"):
        manifest.unlink()
    pipeline.ingest_calls.clear()
    report = _scan(pipeline, root, tmp_path)
    assert report.unchanged == 2 and report.deleted == 0 and pipeline.ingest_calls == []


def test_corrupt_manifest_is_tolerated(root, tmp_path):
    """An unreadable manifest is ignored rather than failing the round."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    for manifest in (tmp_path / "state").glob("*.json"):
        manifest.write_text("{not json", encoding="utf-8")
    report = _scan(pipeline, root, tmp_path)
    assert report.status == "completed" and report.unchanged == 2


def test_rename_is_counted_but_handled_as_delete_plus_add(root, tmp_path):
    """Same content under a new path: reported as a probable rename, identity not preserved."""
    pipeline = FakePipeline()
    _scan(pipeline, root, tmp_path)
    (root / "a.md").rename(root / "a-renamed.md")
    report = _scan(pipeline, root, tmp_path, max_delete_fraction=1.0)
    assert (report.new, report.deleted, report.probable_renames) == (1, 1, 1)


def test_earlier_spelling_of_the_same_file_is_reused(root, tmp_path, monkeypatch):
    """A document stored under another spelling of the root keeps its identity."""
    monkeypatch.chdir(tmp_path)
    pipeline = FakePipeline()
    relative = ScanSourceConfig(path="kb", dataset_id="ds")
    _scan(pipeline, Path("kb"), tmp_path, source=relative)
    stored = set(pipeline.db)
    pipeline.ingest_calls.clear()
    (root / "a.md").write_text("alpha v2", encoding="utf-8")
    report = _scan(pipeline, root, tmp_path)
    assert set(pipeline.db) == stored and report.deleted == 0
    assert report.changed == 1
    assert pipeline.ingest_calls[0]["override"] in stored


def test_tenant_scope_is_passed_as_the_ingest_caller(root, tmp_path):
    """A configured tenant ingests through the governed caller path."""
    pipeline = FakePipeline()
    source = ScanSourceConfig(path=str(root), dataset_id="ds", tenant_id="tenant_a")
    _scan(pipeline, root, tmp_path, source=source)
    assert {c["caller"].tenant_id for c in pipeline.ingest_calls} == {"tenant_a"}
    assert {c["caller"].is_privileged for c in pipeline.ingest_calls} == {False}


def test_vision_or_graph_mode_always_defers_to_ingest_file(root, tmp_path):
    """When the stored checksum is not the plain file digest, ingest_file decides."""
    pipeline = FakePipeline(digest_mode=False)
    _scan(pipeline, root, tmp_path)
    pipeline.ingest_calls.clear()
    report = _scan(pipeline, root, tmp_path)
    assert len(pipeline.ingest_calls) == 2 and report.unchanged == 2


def test_database_failure_reports_error_and_releases_the_lock(root, tmp_path):
    """An unexpected failure ends the round as `error` without deleting, and unlocks."""
    pipeline = FakePipeline()

    def boom(dataset_id):
        raise ConnectionError("db down")

    pipeline.get_document_checksums = boom
    report = _scan(pipeline, root, tmp_path)
    assert report.status == "error" and pipeline.delete_calls == []
    assert list((tmp_path / "state").glob("*.lock")) == []
