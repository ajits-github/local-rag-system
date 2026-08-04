from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import docx

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument


def _as_utc(value: datetime | None, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class DocxLoader(Loader):
    def load(self, path: Path) -> RawDocument:
        document = docx.Document(str(path))
        content = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
        core = document.core_properties
        fallback_created, fallback_modified = file_timestamps(path)

        created_at = _as_utc(core.created, fallback_created)
        last_modified = _as_utc(core.modified, fallback_modified)

        return RawDocument(
            content=content,
            source=str(path),
            source_type="docx",
            title=core.title or path.stem,
            author=core.author or None,
            created_at=created_at,
            last_modified=last_modified,
            language=core.language or None,
        )
