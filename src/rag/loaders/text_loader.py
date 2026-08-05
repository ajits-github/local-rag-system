from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument

# Matches a leading YAML front-matter block delimited by --- lines, as used
# by the TechFusion knowledge base (title/owner/department/last_reviewed/tags).
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n?", re.DOTALL)


def _split_frontmatter(raw: str) -> tuple[dict | None, str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return None, raw
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None, raw
    if not isinstance(frontmatter, dict):
        return None, raw
    return frontmatter, raw[match.end():]


def _parse_date(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class TextLoader(Loader):
    """Handles both plain text and Markdown.

    For Markdown, a leading YAML front-matter block (--- ... ---) is parsed
    for title/owner/last_reviewed and stripped from the chunked content;
    plain .txt files are read as-is with no front-matter handling.
    """

    def load(self, path: Path) -> RawDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        created_at, last_modified = file_timestamps(path)
        source_type = "markdown" if path.suffix.lower() == ".md" else "text"

        title = path.stem
        author = None
        content = raw

        if source_type == "markdown":
            frontmatter, content = _split_frontmatter(raw)
            if frontmatter:
                title = frontmatter.get("title") or title
                author = frontmatter.get("owner")
                last_modified = _parse_date(frontmatter.get("last_reviewed")) or last_modified

        return RawDocument(
            content=content,
            source=str(path),
            source_type=source_type,
            title=title,
            author=author,
            created_at=created_at,
            last_modified=last_modified,
        )
