"""Opt-in scheduled scan that keeps local directories in sync with their datasets.

Run with `python -m rag.ingestion.scheduler [--once]`. Each round enumerates
every configured directory, ingests new and changed files through
`IngestionPipeline.ingest_file`, and deletes documents whose files are gone
only after a complete, plausible enumeration (see `scan_source`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.config import AppConfig, ScanSourceConfig, ScheduledScanConfig, load_config
from rag.ingestion.governance import IngestCallerContext
from rag.ingestion.pipeline import (
    IngestionPipeline,
    _source_is_under_root,
    deletion_blocked_reason,
    discover_files,
    file_category,
    resolved_source_path,
)
from rag.observability import metrics as observability_metrics

logger = logging.getLogger(__name__)

ScanStatus = Literal["completed", "partial", "locked", "root_unavailable", "error"]

_MAX_REPORTED_ERRORS = 20

# One in-process lock per source, so overlapping triggers in one process
# never even reach the cross-process lock file.
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class FileState(BaseModel):
    """What the last scan recorded for one file (fast-path hint only).

    `ino` makes an atomic write-then-rename replacement look changed even
    when it keeps the old size and modification time.
    """

    size: int
    mtime_ns: int
    ino: int = 0
    sha256: str


class ScanReport(BaseModel):
    """Outcome of one scan round for one source.

    Attributes
    ----------
    status : str
        `completed`: everything ingested and deletions reconciled.
        `partial`: some files failed, or deletion reconciliation was
        withheld (see `deletion_blocked_reason`). `locked`: another scan
        of this source is running; nothing was done. `root_unavailable`:
        the directory is missing or unreadable; nothing was done.
        `error`: an unexpected failure (e.g. the database) stopped the
        round before deletion reconciliation.
    deletion_blocked_reason : str or None
        Why deletions were withheld this round, if they were.
    probable_renames : int
        Removed documents whose content reappeared under a new path this
        round. They are still handled as delete plus add.
    """

    source_path: str
    dataset_id: str
    status: ScanStatus
    started_at: str
    duration_ms: float = 0.0
    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deferred: int = 0
    failed: int = 0
    ignored: int = 0
    deleted: int = 0
    deletion_candidates: int = 0
    probable_renames: int = 0
    deletion_blocked_reason: str | None = None
    errors: list[str] = Field(default_factory=list)


def _source_key(source: ScanSourceConfig) -> str:
    """Stable file-name-safe key for one (path, dataset) source."""
    digest = hashlib.sha256(f"{source.path}\0{source.dataset_id}".encode()).hexdigest()[:16]
    return f"{source.dataset_id}-{digest}"


def _file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of `path`'s bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _acquire_lock_file(path: Path, stale_seconds: int) -> bool:
    """Create `path` exclusively; take over a lock older than `stale_seconds` once."""
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if attempt == 0 and age > stale_seconds:
                logger.warning("ingestion_scan_stale_lock_removed", extra={"lock": str(path)})
                path.unlink(missing_ok=True)
                continue
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "acquired_at": datetime.now(UTC).isoformat()}, handle)
        return True
    return False


def _touch(path: Path) -> None:
    """Refresh the lock file's mtime so a long scan is not mistaken for a dead one."""
    try:
        os.utime(path)
    except OSError:
        logger.warning("ingestion_scan_lock_heartbeat_failed", extra={"lock": str(path)})


def _load_manifest(path: Path) -> dict[str, FileState]:
    """Read the per-file fast-path hints; a missing or corrupt manifest yields none."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {source: FileState(**state) for source, state in raw.get("files", {}).items()}
    except (OSError, ValueError, TypeError):
        logger.warning("ingestion_scan_manifest_unreadable", extra={"manifest": str(path)})
        return {}


def _write_manifest(path: Path, report: ScanReport, files: dict[str, FileState]) -> None:
    """Atomically replace the manifest with this round's report and file hints."""
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    payload = {
        "last_report": report.model_dump(),
        "last_completed_at": report.started_at
        if report.status == "completed"
        else previous.get("last_completed_at"),
        "files": {source: state.model_dump() for source, state in sorted(files.items())},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def scan_source(
    pipeline: IngestionPipeline,
    source: ScanSourceConfig,
    scan_config: ScheduledScanConfig,
    supported_extensions: list[str],
    *,
    clock: Any = time.time,
) -> ScanReport:
    """Run one scan round for `source`: ingest new/changed files, then reconcile deletions.

    Deletions are applied only when the directory was enumerated without
    any error, at least one eligible file was found, and the round would
    not remove more than `scan_config.max_delete_fraction` of the source's
    known documents. A file that fails to ingest or is still settling is
    treated as present, so it is never deleted.

    Parameters
    ----------
    pipeline : IngestionPipeline
        Pipeline used for ingestion and deletion.
    source : ScanSourceConfig
        Directory, dataset, and optional tenant scope.
    scan_config : ScheduledScanConfig
        Shared scheduler settings.
    supported_extensions : list[str]
        Default extensions when `source.extensions` is unset.
    clock : callable, optional
        Returns the current time in seconds; injectable for tests.

    Returns
    -------
    ScanReport
        The round's counts and status. Also written to the source's
        manifest unless the round was `locked` or `root_unavailable`.
    """
    started = time.perf_counter()
    report = ScanReport(
        source_path=source.path,
        dataset_id=source.dataset_id,
        status="completed",
        started_at=datetime.now(UTC).isoformat(),
    )
    key = _source_key(source)
    state_dir = Path(scan_config.state_dir)
    lock_path = state_dir / f"{key}.lock"

    with _PROCESS_LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(key, threading.Lock())
    if not process_lock.acquire(blocking=False):
        return _finish(report.model_copy(update={"status": "locked"}), started)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        if not _acquire_lock_file(lock_path, scan_config.lock_stale_seconds):
            return _finish(report.model_copy(update={"status": "locked"}), started)
        try:
            report = _scan_locked(
                pipeline, source, scan_config, supported_extensions, report, lock_path, clock
            )
        except Exception as exc:
            logger.exception("ingestion_scan_failed", extra={"source_path": source.path})
            report = report.model_copy(
                update={"status": "error", "errors": [*report.errors, f"{type(exc).__name__}"]}
            )
        finally:
            lock_path.unlink(missing_ok=True)
    finally:
        process_lock.release()
    return _finish(report, started)


def _finish(report: ScanReport, started: float) -> ScanReport:
    """Stamp the duration, log the report, and record metrics."""
    report = report.model_copy(update={"duration_ms": (time.perf_counter() - started) * 1000})
    log = logger.warning if report.status in ("error", "root_unavailable") else logger.info
    log("ingestion_scan_completed", extra=report.model_dump(exclude={"errors"}))
    observability_metrics.observe_ingestion_scan(
        report.status,
        report.duration_ms / 1000,
        {
            "new": report.new,
            "changed": report.changed,
            "unchanged": report.unchanged,
            "deferred": report.deferred,
            "failed": report.failed,
            "deleted": report.deleted,
        },
    )
    return report


def _scan_locked(
    pipeline: IngestionPipeline,
    source: ScanSourceConfig,
    scan_config: ScheduledScanConfig,
    supported_extensions: list[str],
    report: ScanReport,
    lock_path: Path,
    clock: Any,
) -> ScanReport:
    """Do `scan_source`'s work while holding both locks."""
    root = Path(source.path)
    try:
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        with os.scandir(root) as entries:
            next(entries, None)
    except OSError as exc:
        return report.model_copy(
            update={"status": "root_unavailable", "errors": [f"{type(exc).__name__}: {root}"]}
        )

    manifest_path = Path(scan_config.state_dir) / f"{_source_key(source)}.json"
    manifest = _load_manifest(manifest_path)
    db_checksums = pipeline.vectorstore.get_document_checksums(source.dataset_id)
    existing_under_root = {s for s in db_checksums if _source_is_under_root(s, root)}
    # Reuse an earlier spelling of the same file, so identity never churns.
    spelling_by_resolved = {resolved_source_path(s): s for s in existing_under_root}
    extensions = {ext.lower() for ext in (source.extensions or supported_extensions)}
    caller = (
        IngestCallerContext(tenant_id=source.tenant_id) if source.tenant_id is not None else None
    )

    discovery = discover_files(root, extensions)
    errors: list[str] = []
    counts = {"new": 0, "changed": 0, "unchanged": 0, "deferred": 0, "failed": 0}
    counts["ignored"] = discovery.ignored
    present: set[str] = set()
    new_digests: dict[str, str] = {}
    files_state: dict[str, FileState] = {}

    for file_path in discovery.files:
        persisted = spelling_by_resolved.get(resolved_source_path(str(file_path)), str(file_path))
        present.add(persisted)
        try:
            outcome, state = _process_file(
                pipeline,
                file_path,
                persisted,
                source,
                scan_config,
                caller,
                db_checksums,
                manifest,
                clock,
            )
        except Exception as exc:
            counts["failed"] += 1
            errors.append(f"{type(exc).__name__}: {persisted}")
            logger.warning("ingestion_scan_file_failed", extra={"source": persisted}, exc_info=True)
            continue
        finally:
            _touch(lock_path)
        counts[outcome] += 1
        if state is not None:
            files_state[persisted] = state
            if outcome == "new":
                new_digests[persisted] = state.sha256

    missing = existing_under_root - present
    blocked = deletion_blocked_reason(
        discovery.errors,
        len(present),
        len(existing_under_root),
        len(missing),
        scan_config.max_delete_fraction,
    )
    deleted = 0
    if missing and blocked is None:
        # A file may have reappeared since it was enumerated.
        confirmed = {s for s in missing if not Path(s.replace("\\", "/")).exists()}
        deleted = pipeline.delete_sources(source.dataset_id, confirmed)
    renamed_contents = {db_checksums[s] for s in missing} & set(new_digests.values())
    if missing:
        logger.info(
            "ingestion_scan_deletions",
            extra={
                "source_path": source.path,
                "candidates": len(missing),
                "deleted": deleted,
                "blocked_reason": blocked,
            },
        )

    report = report.model_copy(
        update={
            **counts,
            "discovered": len(present),
            "deleted": deleted,
            "deletion_candidates": len(missing),
            "probable_renames": len(renamed_contents),
            "deletion_blocked_reason": blocked,
            "errors": (discovery.errors + errors)[:_MAX_REPORTED_ERRORS],
            "status": "partial" if counts["failed"] or (missing and blocked) else "completed",
        }
    )
    _write_manifest(manifest_path, report, files_state)
    return report


def _process_file(
    pipeline: IngestionPipeline,
    file_path: Path,
    persisted_source: str,
    source: ScanSourceConfig,
    scan_config: ScheduledScanConfig,
    caller: IngestCallerContext | None,
    db_checksums: dict[str, str],
    manifest: dict[str, FileState],
    clock: Any,
) -> tuple[str, FileState | None]:
    """Classify and, when needed, ingest one file; returns its outcome and new state."""
    stat = file_path.stat()
    # `settle_seconds == 0` disables the window; a fresh mtime can sit just ahead of `clock()`.
    if scan_config.settle_seconds > 0 and clock() - stat.st_mtime < scan_config.settle_seconds:
        return "deferred", None
    stored = db_checksums.get(persisted_source)
    hint = manifest.get(persisted_source)
    if (
        pipeline.checksum_is_file_digest
        and hint is not None
        and stored == hint.sha256
        and (hint.size, hint.mtime_ns, hint.ino) == (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    ):
        return "unchanged", hint
    digest = _file_sha256(file_path)
    state = FileState(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ino=stat.st_ino, sha256=digest)
    if pipeline.checksum_is_file_digest and stored == digest:
        return "unchanged", state
    result = pipeline.ingest_file(
        file_path,
        source.dataset_id,
        category=file_category(Path(source.path), file_path),
        caller=caller,
        source_override=persisted_source if persisted_source != str(file_path) else None,
    )
    if not result["changed"]:
        return "unchanged", state
    return ("changed" if stored is not None else "new"), state


def run_round(config: AppConfig, pipeline: IngestionPipeline) -> list[ScanReport]:
    """Scan every configured source once, in order."""
    scan_config = config.ingestion.scheduled_scan
    return [
        scan_source(pipeline, source, scan_config, config.ingestion.supported_extensions)
        for source in scan_config.sources
    ]


def main() -> None:
    """CLI entrypoint: run one round (`--once`) or loop every `interval_seconds`."""
    from rag.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Scheduled local directory ingestion")
    parser.add_argument("--config", default=None, help="Override config/default.yaml")
    parser.add_argument("--once", action="store_true", help="Run one round and exit")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    configure_logging(config.app.log_level)
    scan_config = config.ingestion.scheduled_scan
    if not scan_config.enabled or not scan_config.sources:
        print(
            "ingestion.scheduled_scan is disabled or has no sources; nothing to do.",
            file=sys.stderr,
        )
        sys.exit(2)

    pipeline = IngestionPipeline(config)
    while True:
        reports = run_round(config, pipeline)
        for report in reports:
            print(report.model_dump_json())
        if args.once:
            failed = any(r.status in ("error", "root_unavailable") or r.failed for r in reports)
            sys.exit(1 if failed else 0)
        time.sleep(scan_config.interval_seconds)


if __name__ == "__main__":
    main()
