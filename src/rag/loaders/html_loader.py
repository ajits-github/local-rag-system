"""Loader for `.html`/`.htm` files."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument


class HTMLLoader(Loader):
    """Extracts visible text and metadata from an HTML document via `BeautifulSoup`."""

    def load(self, path: Path) -> RawDocument:
        """Read `path` and strip it down to its visible text.

        Parameters
        ----------
        path : Path
            Path to the HTML file.

        Returns
        -------
        RawDocument
            Extracted text content, with title/author/language read from
            `<title>`, `<meta name="author">`, and the `<html lang>`
            attribute where present.
        """
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        title_tag = soup.find("title")
        author_tag = soup.find("meta", attrs={"name": "author"})
        author = author_tag.get("content") if author_tag else None
        lang = soup.html.get("lang") if soup.html else None

        content = soup.get_text(separator="\n", strip=True)
        created_at, last_modified = file_timestamps(path)

        return RawDocument(
            content=content,
            source=str(path),
            source_type="html",
            title=(title_tag.get_text(strip=True) if title_tag else path.stem),
            author=(str(author) if author else None),
            created_at=created_at,
            last_modified=last_modified,
            language=(str(lang) if lang else None),
        )
