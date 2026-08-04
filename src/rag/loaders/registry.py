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
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"No loader registered for extension '{suffix}' (file: {path})")
    return loader
