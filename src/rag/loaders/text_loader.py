"""Loader for `.txt` and `.md` files."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument

# Matches a leading YAML front-matter block delimited by --- lines, as used
# by the TechFusion knowledge base (title/owner/department/last_reviewed/tags,
# and -- for the safety_evaluation content set -- tenant_id/allowed_roles/
# classification/status/updated_at/effective_from/document_version/
# supersedes/trust_level/source_type; see RawDocument for what each means).
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n?", re.DOTALL)


def _split_frontmatter(raw: str) -> tuple[dict | None, str]:
    """Split a leading YAML front-matter block off of `raw`, if present.

    Parameters
    ----------
    raw : str
        Full file content, possibly starting with a ``---``-delimited block.

    Returns
    -------
    tuple[dict | None, str]
        The parsed front-matter dict (or ``None`` if absent/invalid) and
        the remaining content with the block stripped.
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return None, raw
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None, raw
    if not isinstance(frontmatter, dict):
        return None, raw
    return frontmatter, raw[match.end() :]


def _parse_date(value: object) -> datetime | None:
    """Parse a front-matter `YYYY-MM-DD` value into a UTC datetime, if valid."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_plain_date(value: object) -> date | None:
    """Parse a front-matter `YYYY-MM-DD` value into a plain `date`, if valid."""
    parsed = _parse_date(value)
    return parsed.date() if parsed else None


def _parse_updated_at(value: object) -> datetime | None:
    """Parse a front-matter `updated_at` (ISO 8601, possibly with a UTC offset)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _parse_roles(value: object) -> list[str] | None:
    """Parse a front-matter `allowed_roles` list, if present and well-formed."""
    if not isinstance(value, list):
        return None
    return [str(role) for role in value]


class TextLoader(Loader):
    """Handles both plain text and Markdown.

    For Markdown, a leading YAML front-matter block (--- ... ---) is parsed
    for title/owner/last_reviewed and stripped from the chunked content;
    plain .txt files are read as-is with no front-matter handling.
    """

    def load(self, path: Path) -> RawDocument:
        """Read `path`, stripping and parsing YAML front-matter for Markdown.

        Parameters
        ----------
        path : Path
            Path to the `.txt` or `.md` file.

        Returns
        -------
        RawDocument
            Extracted content (front-matter block removed for Markdown),
            with title/author/last_modified from the front-matter when
            present, falling back to filename/filesystem values otherwise.
        """
        raw = path.read_text(encoding="utf-8", errors="replace")
        created_at, last_modified = file_timestamps(path)
        source_type = "markdown" if path.suffix.lower() == ".md" else "text"

        title = path.stem
        author = None
        content = raw
        governance: dict[str, Any] = {}

        if source_type == "markdown":
            frontmatter, content = _split_frontmatter(raw)
            if frontmatter:
                title = frontmatter.get("title") or title
                author = frontmatter.get("owner")
                last_modified = (
                    _parse_date(frontmatter.get("last_reviewed"))
                    or _parse_updated_at(frontmatter.get("updated_at"))
                    or last_modified
                )
                governance = {
                    "tenant_id": frontmatter.get("tenant_id"),
                    "allowed_roles": _parse_roles(frontmatter.get("allowed_roles")),
                    "classification": frontmatter.get("classification"),
                    "status": frontmatter.get("status"),
                    "document_version": (
                        str(frontmatter["document_version"])
                        if frontmatter.get("document_version") is not None
                        else None
                    ),
                    "effective_from": _parse_plain_date(frontmatter.get("effective_from")),
                    "trust_level": frontmatter.get("trust_level"),
                    "doc_source_type": frontmatter.get("source_type"),
                    "supersedes_source": frontmatter.get("supersedes"),
                }

        return RawDocument(
            content=content,
            source=str(path),
            source_type=source_type,
            title=title,
            author=author,
            created_at=created_at,
            last_modified=last_modified,
            **governance,
        )
