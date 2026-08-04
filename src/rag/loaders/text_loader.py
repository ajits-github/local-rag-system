from __future__ import annotations

from pathlib import Path

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument


class TextLoader(Loader):
    """Handles both plain text and Markdown — no front-matter parsing for now."""

    def load(self, path: Path) -> RawDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        created_at, last_modified = file_timestamps(path)
        source_type = "markdown" if path.suffix.lower() == ".md" else "text"

        return RawDocument(
            content=content,
            source=str(path),
            source_type=source_type,
            title=path.stem,
            created_at=created_at,
            last_modified=last_modified,
        )
