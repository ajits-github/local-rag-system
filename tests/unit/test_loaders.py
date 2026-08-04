from __future__ import annotations

from pathlib import Path

import docx
import pytest
from pypdf import PdfWriter

from rag.loaders.docx_loader import DocxLoader
from rag.loaders.html_loader import HTMLLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.registry import get_loader
from rag.loaders.text_loader import TextLoader


def test_text_loader_reads_plain_text(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.content == "hello world"
    assert doc.source_type == "text"
    assert doc.source == str(path)


def test_text_loader_detects_markdown(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nbody", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.source_type == "markdown"


def test_html_loader_extracts_title_and_text(tmp_path: Path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html lang='en'><head><title>My Page</title></head>"
        "<body><p>Hello there</p></body></html>",
        encoding="utf-8",
    )

    doc = HTMLLoader().load(path)

    assert doc.title == "My Page"
    assert "Hello there" in doc.content
    assert doc.language == "en"


def test_docx_loader_extracts_paragraphs(tmp_path: Path):
    path = tmp_path / "doc.docx"
    document = docx.Document()
    document.add_paragraph("First paragraph")
    document.add_paragraph("Second paragraph")
    document.core_properties.title = "My Doc"
    document.save(str(path))

    doc = DocxLoader().load(path)

    assert "First paragraph" in doc.content
    assert "Second paragraph" in doc.content
    assert doc.title == "My Doc"


def test_pdf_loader_reads_blank_page_without_error(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)

    doc = PDFLoader().load(path)

    assert doc.source_type == "pdf"
    assert doc.content == ""  # blank page has no extractable text


def test_registry_maps_extensions_to_loaders(tmp_path: Path):
    assert isinstance(get_loader(tmp_path / "a.txt"), TextLoader)
    assert isinstance(get_loader(tmp_path / "a.md"), TextLoader)
    assert isinstance(get_loader(tmp_path / "a.html"), HTMLLoader)
    assert isinstance(get_loader(tmp_path / "a.docx"), DocxLoader)
    assert isinstance(get_loader(tmp_path / "a.pdf"), PDFLoader)


def test_registry_raises_for_unsupported_extension(tmp_path: Path):
    with pytest.raises(ValueError, match="No loader registered"):
        get_loader(tmp_path / "a.xyz")
