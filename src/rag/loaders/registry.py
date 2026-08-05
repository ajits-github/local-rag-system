"""Maps a file's extension to the `Loader` that handles it.

Simple enough to live as an inline extension->instance dict rather than
going through `factory.py` (see CLAUDE.md's swap-point convention).
"""

from __future__ import annotations

from pathlib import Path

from rag.loaders.base import Loader
from rag.loaders.docx_loader import DocxLoader
from rag.loaders.html_loader import HTMLLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.text_loader import TextLoader

_LOADERS: dict[str, Loader] = {
    ".pdf": PDFLoader(),
    ".docx": DocxLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
}


def get_loader(path: Path) -> Loader:
    """Return the `Loader` registered for `path`'s file extension.

    Parameters
    ----------
    path : Path
        File whose extension selects the loader.

    Returns
    -------
    Loader
        The loader instance registered for that extension.

    Raises
    ------
    ValueError
        If no loader is registered for `path`'s extension.
    """
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"No loader registered for extension '{suffix}' (file: {path})")
    return loader
