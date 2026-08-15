"""`Chunker` that keeps Markdown tables, fenced code/config, and charts atomic.

Prevents `RecursiveCharacterChunker` from slicing through structured
blocks once a document exceeds `chunk_size`. Only active for
`source_type == "markdown"`; everything else passes straight through to
the composed `RecursiveCharacterChunker`, which also handles every prose
run within a Markdown document. A prose-only Markdown document therefore
produces chunk text identical to `RecursiveCharacterChunker` used
directly.

`split()` walks the document once, line by line, applying these rules in
priority order:

1. An open fence (```` ``` ```` ... ```` ``` ````) always wins first and is
   consumed verbatim — a `|a|b|`-looking line or a `#` comment *inside* a
   code sample is never reinterpreted, because the fence scanner never
   hands those lines to the table/header detectors.
2. Outside a fence, `#`, `|`, ```` ``` ````, and a standalone `![...](...)`
   image line are mutually exclusive leading tokens, so header/table-start/
   fence-start/image-line detection never genuinely competes for the same
   line.
3. The only real ambiguity is chart-vs-plain-fence, and image-vs-plain-
   image: a ```` ```text ```` fence only becomes `content_type="chart"`,
   and a standalone image line only picks up a `content_type="image"`
   caption, when immediately followed (at most one blank line) by a
   paragraph wholly wrapped in `*...*`/`_..._` emphasis -- both cases share
   the same caption-lookahead helper (`_peek_caption`); otherwise a fence is
   `code`/`configuration` per its language tag, and a bare image line is
   still its own `content_type="image"` span, just without a caption.
4. An image link *embedded inline within a prose paragraph* (not alone on
   its own line) does not create a block boundary — that's still just
   attachment tagging, a property layered onto whichever prose sub-chunk(s)
   literally contain the link, computed after normal prose splitting.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from rag.chunkers.base import Chunker
from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.schemas import ChunkSpan

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{1,}:?\s*\|?\s*$")
_FENCE_START_RE = re.compile(r"^```(\S*)[ \t]*$")
_FENCE_END_RE = re.compile(r"^```[ \t]*$")
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]\[]*)\]\(([^()\s]+)\)")
# A line that is *only* an image reference (as opposed to an image link
# embedded inline within a prose paragraph, which `_tag_attachments` still
# handles). Matches the consistent pattern in the multimodal KB documents:
# an image markdown line on its own, optionally followed by an
# emphasis-wrapped caption paragraph -- the same shape `_consume_fence`
# already recognizes for chart fence+caption.
_IMAGE_LINE_RE = re.compile(r"^!\[([^\]\[]*)\]\(([^()\s]+)\)\s*$")
_ATTACHMENT_EXTENSIONS = {
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".csv",
    ".zip",
}
_CONFIGURATION_LANGUAGES = {"json", "yaml", "yml", "xml"}


def _is_emphasis_wrapped(paragraph: str) -> bool:
    """Return whether `paragraph` is wholly wrapped in `*italic*`/`_italic_` markers."""
    p = paragraph.strip()
    if len(p) < 2:
        return False
    if p.startswith("**") or p.endswith("**") or p.startswith("__") or p.endswith("__"):
        return False
    return (p.startswith("*") and p.endswith("*")) or (p.startswith("_") and p.endswith("_"))


class StructuredMarkdownChunker(Chunker):
    """Splits Markdown into content-aware spans: tables, code/config, charts, and prose.

    Prose (and any oversized structural block) is delegated to
    `RecursiveCharacterChunker` for size-based splitting.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        table_row_group_size: int = 20,
        max_atomic_block_chars: int = 2000,
    ) -> None:
        """Construct the chunker with prose and structural-block size limits.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum characters per prose chunk, by default 500.
        chunk_overlap : int, optional
            Characters shared between consecutive prose chunks, by
            default 50.
        table_row_group_size : int, optional
            Maximum data rows kept in one table chunk before splitting
            into row-groups (each repeating the header row), by default
            20.
        max_atomic_block_chars : int, optional
            Maximum characters a fenced code/config block or a table
            row-group may reach before being split, by default 2000 —
            deliberately decoupled from `chunk_size`, since real fenced
            blocks can exceed a prose-tuned `chunk_size` while still
            being "small" in row/line-count terms.
        """
        self._prose_chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
        self._table_row_group_size = table_row_group_size
        self._max_atomic_block_chars = max_atomic_block_chars

    def split(self, text: str, source_type: str | None = None) -> list[ChunkSpan]:
        """Split `text` into content-aware spans if Markdown, else delegate as-is.

        Parameters
        ----------
        text : str
            Cleaned document text.
        source_type : str | None, optional
            Only `"markdown"` triggers structural parsing; anything else
            (including None) is passed straight to the composed
            `RecursiveCharacterChunker`, by default None.

        Returns
        -------
        list[ChunkSpan]
            Chunk spans, in document order.
        """
        if source_type != "markdown":
            return self._prose_chunker.split(text, source_type)

        lines = text.split("\n")
        n = len(lines)
        spans: list[ChunkSpan] = []
        pending: list[str] = []
        header_stack: list[tuple[int, str]] = []

        def current_section_path() -> str | None:
            return " > ".join(title for _, title in header_stack) if header_stack else None

        def flush_pending() -> None:
            nonlocal pending
            if pending:
                spans.extend(self._flush_prose(pending, current_section_path()))
                pending = []

        i = 0
        while i < n:
            line = lines[i]

            header_match = _HEADER_RE.match(line)
            if header_match:
                pending.append(line)
                level = len(header_match.group(1))
                title = header_match.group(2)
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, title))
                i += 1
                continue

            if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
                flush_pending()
                header_row, sep_row = line, lines[i + 1]
                j = i + 2
                data_rows: list[str] = []
                while j < n and _TABLE_ROW_RE.match(lines[j]):
                    data_rows.append(lines[j])
                    j += 1
                spans.extend(
                    self._split_table(header_row, sep_row, data_rows, current_section_path())
                )
                i = j
                continue

            fence_match = _FENCE_START_RE.match(line)
            if fence_match:
                flush_pending()
                i = self._consume_fence(
                    lines, i, fence_match.group(1) or None, current_section_path(), spans
                )
                continue

            image_match = _IMAGE_LINE_RE.match(line)
            if image_match and self._is_attachment_target(image_match.group(2)):
                flush_pending()
                target = image_match.group(2)
                i = self._consume_image(lines, i, target, current_section_path(), spans)
                continue

            pending.append(line)
            i += 1

        flush_pending()
        return spans

    def _consume_fence(
        self,
        lines: list[str],
        start: int,
        lang: str | None,
        section_path: str | None,
        spans: list[ChunkSpan],
    ) -> int:
        """Consume one fenced block (and its chart caption, if any); return the next index."""
        n = len(lines)
        fence_lines = [lines[start]]
        j = start + 1
        while j < n and not _FENCE_END_RE.match(lines[j]):
            fence_lines.append(lines[j])
            j += 1
        if j < n:
            fence_lines.append(lines[j])
            j += 1
        fence_text = "\n".join(fence_lines)
        after_fence = j

        caption_text, m = self._peek_caption(lines, after_fence)

        if lang and lang.lower() == "text" and caption_text and _is_emphasis_wrapped(caption_text):
            spans.append(
                ChunkSpan(
                    text=f"{fence_text}\n\n{caption_text}",
                    content_type="chart",
                    code_language="text",
                    section_path=section_path,
                )
            )
            return m

        content_type = (
            "configuration" if lang and lang.lower() in _CONFIGURATION_LANGUAGES else "code"
        )
        if len(fence_text) <= self._max_atomic_block_chars:
            spans.append(
                ChunkSpan(
                    text=fence_text,
                    content_type=content_type,
                    code_language=lang,
                    section_path=section_path,
                )
            )
        else:
            for sub in self._prose_chunker.split(fence_text):
                spans.append(
                    sub.model_copy(
                        update={
                            "content_type": content_type,
                            "code_language": lang,
                            "section_path": section_path,
                        }
                    )
                )
        return after_fence

    def _consume_image(
        self,
        lines: list[str],
        start: int,
        target: str,
        section_path: str | None,
        spans: list[ChunkSpan],
    ) -> int:
        """Consume one standalone image line (and its caption, if any); return the next index.

        Mirrors `_consume_fence`'s chart handling: the caption is folded
        into the same span's `text` (not modeled as a separate element) so
        an image and its caption are always retrieved/expanded together.
        """
        image_line = lines[start]
        after_image = start + 1
        caption_text, m = self._peek_caption(lines, after_image)

        if caption_text and _is_emphasis_wrapped(caption_text):
            text = f"{image_line}\n\n{caption_text}"
            next_index = m
        else:
            text = image_line
            next_index = after_image

        spans.append(
            ChunkSpan(
                text=text,
                content_type="image",
                section_path=section_path,
                attachment_name=PurePosixPath(target).name,
                source_anchor=target,
            )
        )
        return next_index

    @staticmethod
    def _peek_caption(lines: list[str], start: int) -> tuple[str, int]:
        """Return the paragraph at `start` (skipping one leading blank line) and its end index.

        Shared by `_consume_fence` (chart captions) and `_consume_image`
        (image captions) -- a pure lookahead that never mutates state.
        Callers decide whether the returned text qualifies as a caption
        (via `_is_emphasis_wrapped`); if it doesn't, they ignore the
        returned end index and let normal prose flow pick the lines back up.

        Parameters
        ----------
        lines : list[str]
            The document's lines.
        start : int
            Index to start looking from (immediately after the fence/image).

        Returns
        -------
        tuple[str, int]
            ``(paragraph_text, index_after_paragraph)``.
        """
        n = len(lines)
        k = start
        if k < n and lines[k].strip() == "":
            k += 1
        caption_lines: list[str] = []
        m = k
        while m < n and lines[m].strip() != "":
            caption_lines.append(lines[m])
            m += 1
        return "\n".join(caption_lines).strip(), m

    @staticmethod
    def _is_attachment_target(target: str) -> bool:
        """Return whether `target` (a markdown link/image URL) points at a local asset file."""
        if target.startswith(("http://", "https://", "mailto:")):
            return False
        return PurePosixPath(target).suffix.lower() in _ATTACHMENT_EXTENSIONS

    def _flush_prose(self, lines: list[str], section_path: str | None) -> list[ChunkSpan]:
        """Split a buffered prose run via the prose chunker, tagging section_path/attachments."""
        joined = "\n".join(lines)
        if not joined.strip():
            return []
        spans = [
            span.model_copy(update={"section_path": section_path})
            for span in self._prose_chunker.split(joined)
        ]
        self._tag_attachments(spans)
        return spans

    def _tag_attachments(self, spans: list[ChunkSpan]) -> None:
        """Tag attachment_name/source_anchor on spans containing a local asset link, in place."""
        for idx, span in enumerate(spans):
            attachment_name: str | None = None
            source_anchor: str | None = None
            for match in _MARKDOWN_LINK_RE.finditer(span.text):
                target = match.group(2)
                if not self._is_attachment_target(target):
                    continue
                attachment_name = PurePosixPath(target).name
                source_anchor = target
            if attachment_name:
                spans[idx] = span.model_copy(
                    update={"attachment_name": attachment_name, "source_anchor": source_anchor}
                )

    def _split_table(
        self,
        header_row: str,
        sep_row: str,
        data_rows: list[str],
        section_path: str | None,
    ) -> list[ChunkSpan]:
        """Group a table's data rows (each group repeating the header) into table span(s)."""
        table_headers = [c.strip() for c in header_row.strip().strip("|").split("|")]

        groups: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for row in data_rows:
            exceeds_chars = (
                bool(current) and current_chars + len(row) > self._max_atomic_block_chars
            )
            exceeds_rows = len(current) >= self._table_row_group_size
            if exceeds_chars or exceeds_rows:
                groups.append(current)
                current, current_chars = [], 0
            current.append(row)
            current_chars += len(row)
        groups.append(current)

        spans = []
        row_offset = 0
        for group in groups:
            source_anchor = None
            if len(groups) > 1:
                source_anchor = f"rows {row_offset + 1}-{row_offset + len(group)}"
            spans.append(
                ChunkSpan(
                    text="\n".join([header_row, sep_row, *group]),
                    content_type="table",
                    table_headers=table_headers,
                    section_path=section_path,
                    source_anchor=source_anchor,
                )
            )
            row_offset += len(group)
        return spans
