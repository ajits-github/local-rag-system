"""Loader for `.docx` files: layout-aware structural extraction.

Serializes headings/tables/images/page breaks into the same
Markdown-equivalent syntax `StructuredMarkdownChunker` already
understands (see that module's docstring and `loaders/pdf_loader.py`'s,
which takes the same approach for PDFs) -- `DocxLoader.load()` still
returns a `RawDocument` with a single `content: str`, unchanged
interface.

Unlike a PDF, a `.docx` body is already a flat, ordered sequence of
paragraphs and tables (`document.element.body`'s children) -- no
geometry/font-size-based reading-order reconstruction is needed, only a
single linear walk. Three structural signals come from the OOXML tree
rather than `python-docx`'s higher-level (paragraph-text-only) API:

- **Page numbers**: a manual page break (`<w:br w:type="page"/>`) is the
  only reliable page signal `python-docx` exposes -- Word's own
  reflow-based pagination isn't computable without a rendering engine.
  Documents with no manual page breaks get `page=1` throughout (honest:
  reflow position is genuinely unknowable, not silently guessed).
- **Inline images**: found via `.//a:blip` on each paragraph's XML
  (the standard OOXML image-reference element), resolved to raw bytes
  through `document.part.related_parts[rId].blob`.
- **Code/config blocks**: no dedicated Word style is used in this
  project's corpus, so detection is by run-level monospace font name
  (Consolas/Courier New/etc.) rather than `paragraph.style.name` --
  see `_is_monospace_paragraph`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from rag.loaders.base import Loader, file_timestamps, resolve_image_asset
from rag.loaders.markdown_render import render_table, sniff_code_language
from rag.schemas import RawDocument

_FIGURE_CAPTION_RE = re.compile(r"^(figure|table)\s+\d+[.:)]", re.IGNORECASE)
_MONOSPACE_FONTS = {"consolas", "courier new", "courier", "lucida console", "monaco", "menlo"}


def _as_utc(value: datetime | None, fallback: datetime) -> datetime:
    """Coerce a possibly-naive/possibly-missing timestamp to UTC-aware."""
    if value is None:
        return fallback
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _is_monospace_paragraph(paragraph: Paragraph) -> bool:
    """Whether every non-empty run in `paragraph` uses a monospace font.

    The only code/config signal available in this project's DOCX corpus
    (see module docstring); a paragraph with no runs at all is not
    considered monospace.
    """
    runs_with_text = [r for r in paragraph.runs if r.text]
    if not runs_with_text:
        return False
    return all((r.font.name or "").strip().lower() in _MONOSPACE_FONTS for r in runs_with_text)


def _heading_marker(style_name: str) -> str | None:
    """Map a paragraph style name to a heading marker, or `None` if it's not a heading."""
    name = (style_name or "").strip().lower()
    if name == "title":
        return "#"
    if name.startswith("heading"):
        return "##"
    return None


def _has_page_break(paragraph: Paragraph) -> bool:
    """Whether `paragraph` contains an explicit manual page break."""
    for br in paragraph._element.findall(".//" + qn("w:br")):
        if br.get(qn("w:type")) == "page":
            return True
    return False


def _paragraph_images(paragraph: Paragraph) -> list[str]:
    """Return the relationship ids of every embedded image referenced in `paragraph`."""
    rids = []
    for blip in paragraph._element.xpath(".//a:blip"):
        rid = blip.get(qn("r:embed"))
        if rid:
            rids.append(rid)
    return rids


class DocxLoader(Loader):
    """Extracts layout-aware structure from a `.docx` via `python-docx`."""

    def load(self, path: Path) -> RawDocument:
        """Read `path` and serialize its layout structure into Markdown-equivalent text.

        Parameters
        ----------
        path : Path
            Path to the `.docx` file.

        Returns
        -------
        RawDocument
            Extracted content (headings/tables/images/page markers as
            Markdown-equivalent syntax) with title/author/language/
            timestamps from the document's core properties, falling back
            to the filename/filesystem timestamps where those properties
            are unset.
        """
        document = docx.Document(str(path))
        core = document.core_properties
        fallback_created, fallback_modified = file_timestamps(path)

        blocks = self._render_body(document, path)
        content = "\n\n".join(blocks)

        return RawDocument(
            content=content,
            source=str(path),
            source_type="docx",
            title=core.title or path.stem,
            author=core.author or None,
            created_at=_as_utc(core.created, fallback_created),
            last_modified=_as_utc(core.modified, fallback_modified),
            language=core.language or None,
        )

    def _render_body(self, document: DocxDocument, path: Path) -> list[str]:
        """Walk the document body once, emitting Markdown-equivalent blocks in order.

        Parameters
        ----------
        document : DocxDocument
            The opened document.
        path : Path
            The document's own path (for resolving/writing image assets).

        Returns
        -------
        list[str]
            Ordered block strings, with `<!--page:N-->` sentinels
            inserted after any paragraph containing a manual page break.
        """
        blocks: list[str] = ["<!--page:1-->"]
        current_page = 1
        doc_image_index = 0
        code_buffer: list[str] = []

        def flush_code() -> None:
            nonlocal code_buffer
            if code_buffer:
                text = "\n".join(code_buffer)
                lang = sniff_code_language(text)
                fence_lang = lang or ""
                blocks.append(f"```{fence_lang}\n{text}\n```")
                code_buffer = []

        children = list(document.element.body.iterchildren())
        i = 0
        while i < len(children):
            child = children[i]
            if child.tag == qn("w:tbl"):
                flush_code()
                table = DocxTable(child, document)
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                blocks.append(render_table(rows))
                i += 1
                continue
            if child.tag != qn("w:p"):
                i += 1
                continue

            paragraph = Paragraph(child, document)
            image_rids = _paragraph_images(paragraph)
            if image_rids:
                flush_code()
                for rid in image_rids:
                    part = document.part.related_parts[rid]
                    ext = (
                        "." + part.content_type.split("/")[-1]
                        if "/" in part.content_type
                        else ".png"
                    )
                    ext = {".jpeg": ".jpg"}.get(ext, ext)

                    def bytes_factory(data: bytes = part.blob, ext: str = ext) -> tuple[bytes, str]:
                        return data, ext

                    target = resolve_image_asset(path, doc_image_index, bytes_factory)
                    caption = None
                    if i + 1 < len(children) and children[i + 1].tag == qn("w:p"):
                        next_text = Paragraph(children[i + 1], document).text.strip()
                        if _FIGURE_CAPTION_RE.match(next_text):
                            caption = next_text
                            i += 1
                    block = f"![Figure {doc_image_index + 1}]({target})"
                    if caption:
                        block += f"\n\n*{caption}*"
                    blocks.append(block)
                    doc_image_index += 1
                i += 1
                if _has_page_break(paragraph):
                    current_page += 1
                    blocks.append(f"<!--page:{current_page}-->")
                continue

            text = paragraph.text.strip()
            if text:
                if _is_monospace_paragraph(paragraph):
                    code_buffer.append(text)
                else:
                    flush_code()
                    marker = _heading_marker(paragraph.style.name if paragraph.style else "")
                    blocks.append(f"{marker} {text}" if marker else text)
            if _has_page_break(paragraph):
                flush_code()
                current_page += 1
                blocks.append(f"<!--page:{current_page}-->")
            i += 1

        flush_code()
        return blocks
