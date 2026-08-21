"""Loader interface: every source-type loader returns a common RawDocument."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rag.schemas import RawDocument


class Loader(ABC):
    """Reads one source-type's files from disk into a `RawDocument`."""

    @abstractmethod
    def load(self, path: Path) -> RawDocument:
        """Read `path` from disk and return its content + base metadata.

        Parameters
        ----------
        path : Path
            Path to the file to load.

        Returns
        -------
        RawDocument
            The extracted content and metadata, before cleaning/chunking.
        """


def file_timestamps(path: Path) -> tuple[datetime, datetime]:
    """Best-effort (created_at, last_modified) from filesystem stat.

    st_ctime is creation time on Windows but metadata-change time on
    POSIX; it's used here only as a fallback when a document format
    doesn't carry its own creation timestamp (e.g. plain text/HTML).

    Parameters
    ----------
    path : Path
        Path whose filesystem stat to read.

    Returns
    -------
    tuple[datetime, datetime]
        ``(created_at, last_modified)``, both timezone-aware UTC.
    """
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime, tz=UTC)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    return created_at, last_modified


_IMAGE_ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg")
_ASSET_MATCH_STOPWORDS = {
    "view",
    "figure",
    "fig",
    "image",
    "review",
    "report",
    "guide",
    "analysis",
    "brief",
    "notes",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem_words(stem: str) -> set[str]:
    """Lowercased, stopword-filtered word set from a filename stem, for asset-group matching."""
    return {w for w in _WORD_RE.findall(stem.lower()) if w not in _ASSET_MATCH_STOPWORDS}


def _matching_assets(document_path: Path, candidates: list[Path]) -> list[Path]:
    """Narrow `candidates` (a shared `assets/` folder) to the ones belonging to `document_path`.

    A category folder can hold more than one source document, each with
    its own `assets/` subfolder shared between them (e.g. three review
    documents all under `operations/`, each contributing two images to
    one `operations/assets/`) -- plain positional pairing across the
    whole folder would silently hand every document the *first*
    document's images. Instead, group candidates by whether their own
    filename shares a (non-stopword) word with `document_path`'s
    filename -- true for this project's asset-naming convention (e.g.
    `capacity-view-01.png` for `ingestion-capacity-analysis.docx`, via
    the shared word "capacity"). Falls back to every candidate, in sorted
    order, if no candidate matches any word (the common case: a folder
    holding only one document's assets, or a naming convention this
    heuristic can't recognize -- better than resolving zero images).

    Parameters
    ----------
    document_path : Path
        Path to the PDF/DOCX file being loaded.
    candidates : list[Path]
        Every image file in the sibling `assets/` directory, sorted.

    Returns
    -------
    list[Path]
        The subset of `candidates` judged to belong to this document, in
        sorted order.
    """
    doc_words = _stem_words(document_path.stem)
    matched = [c for c in candidates if doc_words & _stem_words(c.stem)]
    return matched or candidates


def resolve_image_asset(
    document_path: Path,
    image_index: int,
    image_bytes_factory: Callable[[], tuple[bytes, str]],
) -> str:
    """Resolve a Markdown-emittable relative path for one embedded PDF/DOCX image.

    Shared by `PDFLoader`/`DocxLoader`: prefers an existing sibling
    `assets/` directory next to `document_path` (see `_matching_assets`
    for how candidates are narrowed to this document when the folder is
    shared, then paired positionally with embedded images in document
    order), the convention this project's evaluation corpora already use
    for pre-supplied figures. Falls back to writing the image's own bytes
    (from `image_bytes_factory`, called only on a fallback -- so a
    PDF/DOCX with no pre-existing `assets/` folder still ingests
    successfully) into a new `assets/` directory using a generic
    `<document-stem>-figure-{n:02d}<ext>` name.

    Called on every ingestion of the document (the checksum-unchanged
    short-circuit in `IngestionPipeline.ingest_file` happens after
    `Loader.load()` runs), so the fallback write is idempotent by
    construction: same index always resolves to the same deterministic
    filename with the same bytes, just re-written.

    Parameters
    ----------
    document_path : Path
        Path to the PDF/DOCX file the image was found in.
    image_index : int
        0-based position of this image among all images in the document,
        in document order.
    image_bytes_factory : Callable[[], tuple[bytes, str]]
        Called only on a fallback (no matching pre-existing asset);
        returns `(raw_bytes, file_extension)` for the embedded image.

    Returns
    -------
    str
        A POSIX-style path relative to `document_path`'s own directory
        (e.g. `"assets/architecture-view-01.png"`), suitable as a
        `ChunkSpan.source_anchor`.
    """
    assets_dir = document_path.parent / "assets"
    if assets_dir.is_dir():
        existing = sorted(
            p for p in assets_dir.iterdir() if p.suffix.lower() in _IMAGE_ASSET_EXTENSIONS
        )
        matched = _matching_assets(document_path, existing)
        if image_index < len(matched):
            return f"assets/{matched[image_index].name}"
    assets_dir.mkdir(exist_ok=True)
    data, ext = image_bytes_factory()
    name = f"{document_path.stem}-figure-{image_index + 1:02d}{ext}"
    (assets_dir / name).write_bytes(data)
    return f"assets/{name}"
