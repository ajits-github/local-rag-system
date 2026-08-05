"""Loader for `.pdf` files."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.loaders.base import Loader, file_timestamps
from rag.schemas import RawDocument


class PDFLoader(Loader):
    """Extracts text and metadata from a PDF via `pypdf`."""

    def load(self, path: Path) -> RawDocument:
        """Read `path` and concatenate each page's extracted text.

        Parameters
        ----------
        path : Path
            Path to the PDF file.

        Returns
        -------
        RawDocument
            Extracted content with title/author from the PDF's document
            info dictionary, falling back to the filename/filesystem
            timestamps where the PDF has no metadata.
        """
        reader = PdfReader(str(path))
        content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        meta = reader.metadata
        created_at, last_modified = file_timestamps(path)
        return RawDocument(
            content=content,
            source=str(path),
            source_type="pdf",
            title=(meta.title if meta and meta.title else path.stem),
            author=(meta.author if meta and meta.author else None),
            created_at=created_at,
            last_modified=last_modified,
        )
