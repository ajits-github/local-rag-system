"""Loader interface: every source-type loader returns a common RawDocument."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from rag.schemas import RawDocument


class Loader(ABC):
    @abstractmethod
    def load(self, path: Path) -> RawDocument:
        """Read `path` from disk and return its content + base metadata."""


def file_timestamps(path: Path) -> tuple[datetime, datetime]:
    """Best-effort (created_at, last_modified) from filesystem stat.

    st_ctime is creation time on Windows but metadata-change time on
    POSIX; it's used here only as a fallback when a document format
    doesn't carry its own creation timestamp (e.g. plain text/HTML).
    """
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return created_at, last_modified
