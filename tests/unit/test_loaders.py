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
    """A plain .txt file is read as-is with source_type "text"."""
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.content == "hello world"
    assert doc.source_type == "text"
    assert doc.source == str(path)


def test_text_loader_parses_yaml_frontmatter(tmp_path: Path):
    """YAML front-matter populates title/author/last_modified and is stripped from content."""
    path = tmp_path / "policy.md"
    path.write_text(
        "---\n"
        'title: "Access Control Policy"\n'
        'owner: "Security Engineering"\n'
        'last_reviewed: "2026-07-15"\n'
        'tags: ["access", "rbac"]\n'
        "---\n\n"
        "# Access Control Policy\n\nBody text here.",
        encoding="utf-8",
    )

    doc = TextLoader().load(path)

    assert doc.title == "Access Control Policy"
    assert doc.author == "Security Engineering"
    assert doc.last_modified.strftime("%Y-%m-%d") == "2026-07-15"
    assert "---" not in doc.content
    assert "Body text here." in doc.content


def test_text_loader_parses_governance_frontmatter(tmp_path: Path):
    """Safety/freshness front-matter fields populate RawDocument's new governance fields."""
    path = tmp_path / "retention-policy-v2.md"
    path.write_text(
        "---\n"
        'title: "Tenant Alpha Retention Policy v2"\n'
        'tenant_id: "tenant_alpha"\n'
        "allowed_roles:\n"
        '  - "tenant_alpha_operator"\n'
        '  - "tenant_alpha_admin"\n'
        'classification: "internal"\n'
        'status: "active"\n'
        'updated_at: "2026-06-01T10:00:00+02:00"\n'
        'effective_from: "2026-06-01"\n'
        'document_version: "2.0"\n'
        "supersedes: retention-policy-v1.md\n"
        'trust_level: "authoritative"\n'
        'source_type: "controlled_internal"\n'
        "---\n\n"
        "# Retention\n\nBody text here.",
        encoding="utf-8",
    )

    doc = TextLoader().load(path)

    assert doc.tenant_id == "tenant_alpha"
    assert doc.allowed_roles == ["tenant_alpha_operator", "tenant_alpha_admin"]
    assert doc.classification == "internal"
    assert doc.status == "active"
    assert doc.effective_from.isoformat() == "2026-06-01"
    assert doc.document_version == "2.0"
    assert doc.supersedes_source == "retention-policy-v1.md"
    assert doc.trust_level == "authoritative"
    assert doc.doc_source_type == "controlled_internal"
    assert doc.source_type == "markdown"  # unaffected: the loader-type field, not doc_source_type
    assert doc.last_modified.isoformat() == "2026-06-01T10:00:00+02:00"


def test_text_loader_governance_fields_none_without_frontmatter(tmp_path: Path):
    """A document with no governance front matter leaves every new field unset."""
    path = tmp_path / "plain.md"
    path.write_text("# Title\n\nbody, no frontmatter here", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.tenant_id is None
    assert doc.allowed_roles is None
    assert doc.trust_level is None
    assert doc.doc_source_type is None
    assert doc.supersedes_source is None


def test_text_loader_handles_markdown_without_frontmatter(tmp_path: Path):
    """Markdown without a front-matter block falls back to filename/raw content."""
    path = tmp_path / "plain.md"
    path.write_text("# Title\n\nbody, no frontmatter here", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.title == "plain"  # falls back to filename stem
    assert doc.author is None
    assert doc.content == "# Title\n\nbody, no frontmatter here"


def test_text_loader_detects_markdown(tmp_path: Path):
    """A .md extension sets source_type to "markdown"."""
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nbody", encoding="utf-8")

    doc = TextLoader().load(path)

    assert doc.source_type == "markdown"


def test_html_loader_extracts_title_and_text(tmp_path: Path):
    """HTML title/lang attributes and visible body text are extracted."""
    path = tmp_path / "page.html"
    path.write_text(
        "<html lang='en'><head><title>My Page</title></head><body><p>Hello there</p></body></html>",
        encoding="utf-8",
    )

    doc = HTMLLoader().load(path)

    assert doc.title == "My Page"
    assert "Hello there" in doc.content
    assert doc.language == "en"


def test_docx_loader_extracts_paragraphs(tmp_path: Path):
    """Paragraph text and the title core property are extracted from a .docx."""
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
    """A PDF with no extractable text loads with empty content, not an error."""
    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)

    doc = PDFLoader().load(path)

    assert doc.source_type == "pdf"
    assert doc.content == ""  # blank page has no extractable text


def test_registry_maps_extensions_to_loaders(tmp_path: Path):
    """get_loader returns the right Loader subclass per file extension."""
    assert isinstance(get_loader(tmp_path / "a.txt"), TextLoader)
    assert isinstance(get_loader(tmp_path / "a.md"), TextLoader)
    assert isinstance(get_loader(tmp_path / "a.html"), HTMLLoader)
    assert isinstance(get_loader(tmp_path / "a.docx"), DocxLoader)
    assert isinstance(get_loader(tmp_path / "a.pdf"), PDFLoader)


def test_registry_raises_for_unsupported_extension(tmp_path: Path):
    """get_loader raises ValueError for an unregistered extension."""
    with pytest.raises(ValueError, match="No loader registered"):
        get_loader(tmp_path / "a.xyz")
