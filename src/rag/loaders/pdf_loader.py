"""Loader for `.pdf` files: layout-aware structural extraction.

Serializes headings/tables/images/page numbers into the same
Markdown-equivalent syntax `StructuredMarkdownChunker` already
understands (see that module's docstring) instead of building a
parallel element/chunk model -- `PDFLoader.load()` still returns a
`RawDocument` with a single `content: str`, unchanged interface.

`pdfplumber` (`config.ingestion.layout_parsing.pdf_parser`, the only
parser implemented so far) supplies word-level font sizes (heading
detection), table extraction, and page/image geometry. `pypdf` still
supplies document-info metadata (title/author, unchanged from before
this milestone) and raw embedded-image bytes for the fallback asset
path (see `loaders.base.resolve_image_asset`).

Heading detection is a font-size-ratio heuristic (a line whose dominant
word size is `>= body_size * heading_font_size_ratio`, short enough to
plausibly be a heading, and not already claimed by a table), not a real
layout model -- documented, not hidden. Two ratio tiers
(`heading_font_size_ratio` / `title_font_size_ratio`) give a title level
(`#`) and a section level (`##`); anything deeper collapses to `##`,
so `section_path` for a PDF is at most two levels deep even where the
source visually has more.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

from rag.loaders.base import Loader, file_timestamps, resolve_image_asset
from rag.loaders.markdown_render import render_table, sniff_code_language
from rag.schemas import RawDocument

_FIGURE_CAPTION_RE = re.compile(r"^(figure|table)\s+\d+[.:)]", re.IGNORECASE)
_LINE_GAP_EPS = 3.0  # points; words within this vertical distance are the same line
# A line-to-line gap this much larger than the page's median line gap starts a new paragraph.
_PARAGRAPH_GAP_RATIO = 1.8
_MONOSPACE_FONT_NAMES = {"courier", "consolas", "lucidaconsole", "monaco", "menlo", "couriernew"}


class PDFLoader(Loader):
    """Extracts layout-aware structure from a PDF via `pdfplumber` + `pypdf`."""

    def __init__(
        self,
        heading_font_size_ratio: float = 1.15,
        title_font_size_ratio: float = 1.8,
        max_heading_words: int = 12,
    ) -> None:
        """Store the font-size heuristics used to detect headings.

        Parameters
        ----------
        heading_font_size_ratio : float, optional
            Minimum word-size-to-body-size ratio for a short line to
            count as a section heading (`##`), by default 1.15.
        title_font_size_ratio : float, optional
            Minimum ratio for a title-level heading (`#`), by default 1.8.
        max_heading_words : int, optional
            A candidate heading line longer than this many words is
            treated as prose instead (large-font pull-quotes/labels are
            not headings), by default 12.
        """
        self._heading_font_size_ratio = heading_font_size_ratio
        self._title_font_size_ratio = title_font_size_ratio
        self._max_heading_words = max_heading_words

    def load(self, path: Path) -> RawDocument:
        """Read `path` and serialize its layout structure into Markdown-equivalent text.

        Parameters
        ----------
        path : Path
            Path to the PDF file.

        Returns
        -------
        RawDocument
            Extracted content (headings/tables/images/page markers as
            Markdown-equivalent syntax) with title/author from the PDF's
            document info dictionary, falling back to the filename/
            filesystem timestamps where the PDF has no metadata.
        """
        reader = PdfReader(str(path))
        meta = reader.metadata
        created_at, last_modified = file_timestamps(path)

        page_blocks: list[str] = []
        doc_image_index = 0
        with pdfplumber.open(str(path)) as pdf:
            body_size = self._estimate_body_font_size(pdf)
            for page_num, page in enumerate(pdf.pages, start=1):
                pypdf_images = self._pypdf_page_images(reader, page_num)
                rendered, doc_image_index = self._render_page(
                    page, path, page_num, body_size, pypdf_images, doc_image_index
                )
                page_blocks.append(f"<!--page:{page_num}-->")
                page_blocks.extend(rendered)

        content = "\n\n".join(
            block for block in page_blocks if block.strip() or block.startswith("<!--")
        )
        return RawDocument(
            content=content,
            source=str(path),
            source_type="pdf",
            title=(meta.title if meta and meta.title else path.stem),
            author=(meta.author if meta and meta.author else None),
            created_at=created_at,
            last_modified=last_modified,
        )

    @staticmethod
    def _pypdf_page_images(reader: PdfReader, page_num: int) -> list[Any]:
        """Return `pypdf`'s embedded images for 1-indexed `page_num`, or `[]` past the last page."""
        if page_num - 1 >= len(reader.pages):
            return []
        return list(reader.pages[page_num - 1].images)

    def _estimate_body_font_size(self, pdf: pdfplumber.PDF) -> float:
        """Estimate the document's dominant (body-text) word font size.

        Parameters
        ----------
        pdf : pdfplumber.PDF
            The open PDF.

        Returns
        -------
        float
            The most common word size across every page, or 10.0 as a
            last-resort default if the PDF has no extractable text.
        """
        from collections import Counter

        sizes: Counter[float] = Counter()
        for page in pdf.pages:
            for word in page.extract_words(extra_attrs=["size"]):
                sizes[round(word["size"], 1)] += 1
        if not sizes:
            return 10.0
        return sizes.most_common(1)[0][0]

    def _render_page(
        self,
        page: pdfplumber.page.Page,
        path: Path,
        page_num: int,
        body_size: float,
        pypdf_images: list[Any],
        doc_image_index: int,
    ) -> tuple[list[str], int]:
        """Render one page's tables/headings/prose/images as ordered Markdown-equivalent blocks.

        Parameters
        ----------
        page : pdfplumber.page.Page
            The page to render.
        path : Path
            The PDF's own path (for resolving/writing image assets).
        page_num : int
            1-indexed page number.
        body_size : float
            Document-wide dominant word font size, from `_estimate_body_font_size`.
        pypdf_images : list
            `pypdf` embedded-image objects for this page, in stream order
            (assumed to correspond 1:1 with `pdfplumber`'s `page.images`
            for these simple, single-column documents -- not a guarantee
            for arbitrarily complex PDFs).
        doc_image_index : int
            Running 0-based image count across the whole document so far.

        Returns
        -------
        tuple[list[str], int]
            Ordered list of block strings for this page, and the updated
            `doc_image_index`.
        """
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]

        def in_table(word: dict[str, Any]) -> bool:
            cx, cy = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
            return any(
                x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in table_bboxes
            )

        words = [w for w in page.extract_words(extra_attrs=["size", "fontname"]) if not in_table(w)]
        paragraphs = self._group_paragraphs(words, body_size)

        items: list[tuple[float, str, Any]] = []
        for para in paragraphs:
            items.append((para["top"], "paragraph", para))
        for table in tables:
            items.append((table.bbox[1], "table", table))
        for img in page.images:
            items.append((img["top"], "image", img))
        items.sort(key=lambda t: t[0])

        blocks: list[str] = []
        page_image_counter = 0
        i = 0
        while i < len(items):
            _, kind, payload = items[i]
            if kind == "paragraph":
                blocks.append(self._render_paragraph(payload))
                i += 1
            elif kind == "table":
                blocks.append(self._render_table(payload))
                i += 1
            else:
                consumed = 1
                if i + 1 < len(items) and items[i + 1][1] == "paragraph":
                    next_para = items[i + 1][2]
                    if _FIGURE_CAPTION_RE.match(next_para["text"].strip()):
                        consumed = 2
                caption = items[i + 1][2]["text"].strip() if consumed == 2 else None
                block, doc_image_index = self._render_image(
                    path, doc_image_index, page_image_counter, pypdf_images, caption
                )
                page_image_counter += 1
                blocks.append(block)
                i += consumed
        return blocks, doc_image_index

    def _group_paragraphs(
        self, words: list[dict[str, Any]], body_size: float
    ) -> list[dict[str, Any]]:
        """Group non-table words into lines, then lines into paragraphs/headings.

        Parameters
        ----------
        words : list[dict]
            Non-table words with `text`/`x0`/`x1`/`top`/`bottom`/`size`,
            in `pdfplumber`'s natural (top-to-bottom, left-to-right)
            reading order.
        body_size : float
            Document-wide dominant word font size.

        Returns
        -------
        list[dict]
            One dict per paragraph/heading/code block, sorted by `top`:
            `{"top", "text", "kind"}` where `kind` is one of
            `"prose"`/`"heading1"`/`"heading2"`/`"code"`.
        """
        lines: list[dict[str, Any]] = []
        current: list[dict[str, Any]] | None = None
        for word in words:
            if current and abs(word["top"] - current[0]["top"]) <= _LINE_GAP_EPS:
                current.append(word)
            else:
                if current:
                    lines.append(self._summarize_line(current))
                current = [word]
        if current:
            lines.append(self._summarize_line(current))
        if not lines:
            return []

        gaps = [max(0.0, lines[i]["top"] - lines[i - 1]["bottom"]) for i in range(1, len(lines))]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0

        paragraphs: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []
        code_buffer: list[dict[str, Any]] = []

        def flush() -> None:
            if buffer:
                paragraphs.append(
                    {
                        "top": buffer[0]["top"],
                        "text": " ".join(line["text"] for line in buffer),
                        "kind": "prose",
                    }
                )
                buffer.clear()

        def flush_code() -> None:
            if code_buffer:
                paragraphs.append(
                    {
                        "top": code_buffer[0]["top"],
                        "text": "\n".join(line["text"] for line in code_buffer),
                        "kind": "code",
                    }
                )
                code_buffer.clear()

        for line in lines:
            is_heading = (
                line["size"] >= body_size * self._heading_font_size_ratio
                and len(line["text"].split()) <= self._max_heading_words
                and not line["is_monospace"]
            )
            if is_heading:
                flush()
                flush_code()
                level = 1 if line["size"] >= body_size * self._title_font_size_ratio else 2
                paragraphs.append(
                    {"top": line["top"], "text": line["text"], "kind": f"heading{level}"}
                )
                continue
            if line["is_monospace"]:
                flush()
                code_buffer.append(line)
                continue
            flush_code()
            if buffer and (line["top"] - buffer[-1]["bottom"]) > median_gap * _PARAGRAPH_GAP_RATIO:
                flush()
            buffer.append(line)
        flush()
        flush_code()
        paragraphs.sort(key=lambda p: p["top"])
        return paragraphs

    @staticmethod
    def _summarize_line(words: list[dict[str, Any]]) -> dict[str, Any]:
        """Collapse one line's words into `{text, top, bottom, size, is_monospace}`."""
        from collections import Counter

        sizes = Counter(round(w["size"], 1) for w in words)
        is_monospace = all(
            (w.get("fontname") or "").split("+")[-1].split("-")[0].lower().replace(" ", "")
            in _MONOSPACE_FONT_NAMES
            for w in words
        )
        return {
            "text": " ".join(w["text"] for w in words),
            "top": words[0]["top"],
            "bottom": max(w["bottom"] for w in words),
            "size": sizes.most_common(1)[0][0],
            "is_monospace": is_monospace,
        }

    @staticmethod
    def _render_paragraph(para: dict[str, Any]) -> str:
        """Render one paragraph/heading/code dict as a Markdown-equivalent block."""
        if para["kind"] == "heading1":
            return f"# {para['text']}"
        if para["kind"] == "heading2":
            return f"## {para['text']}"
        if para["kind"] == "code":
            lang = sniff_code_language(para["text"]) or ""
            return f"```{lang}\n{para['text']}\n```"
        return str(para["text"])

    @staticmethod
    def _render_table(table: pdfplumber.table.Table) -> str:
        """Render a `pdfplumber` `Table` as a Markdown pipe table."""
        return render_table(table.extract())

    @staticmethod
    def _render_image(
        path: Path,
        doc_image_index: int,
        page_image_index: int,
        pypdf_images: list[Any],
        caption: str | None,
    ) -> tuple[str, int]:
        """Render one embedded image as a Markdown-equivalent image (+ optional caption) block.

        Parameters
        ----------
        path : Path
            The PDF's own path.
        doc_image_index : int
            0-based running image count across the whole document so far
            (the position `resolve_image_asset` pairs against a sorted
            sibling `assets/` folder).
        page_image_index : int
            0-based image count within this page so far, used to index
            into `pypdf_images` for the byte-extraction fallback (assumes
            `pdfplumber`'s and `pypdf`'s per-page image enumeration order
            match -- true for these simple, single-column documents; see
            the module docstring).
        pypdf_images : list
            This page's `pypdf` embedded-image objects.
        caption : str | None
            Caption text (e.g. "Figure 1. ...") immediately following
            this image, if one was detected by the caller.

        Returns
        -------
        tuple[str, int]
            The rendered block and the incremented `doc_image_index`.
        """

        def bytes_factory() -> tuple[bytes, str]:
            if page_image_index < len(pypdf_images):
                candidate = pypdf_images[page_image_index]
                data: bytes = candidate.data
                ext = Path(candidate.name).suffix or ".png"
                return data, ext
            return b"", ".png"

        target = resolve_image_asset(path, doc_image_index, bytes_factory)
        alt = f"Figure {doc_image_index + 1}"
        block = f"![{alt}]({target})"
        if caption:
            block += f"\n\n*{caption}*"
        return block, doc_image_index + 1
