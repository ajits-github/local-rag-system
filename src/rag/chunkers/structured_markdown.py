"""`Chunker` that keeps Markdown tables, fenced code/config, and charts atomic.

Instead of slicing through them the way `RecursiveCharacterChunker` would
once a document exceeds `chunk_size`. Only active for
`source_type == "markdown"` — everything else passes
straight through to the composed `RecursiveCharacterChunker`, which is
also what handles every prose run *within* a Markdown document. This is
why a prose-only Markdown document (no table/fence syntax) produces
chunk text byte-identical to `RecursiveCharacterChunker` used directly:
the whole document becomes one prose run, flushed through the same
chunker with the same parameters.

`split()` walks the document once, line by line, applying these rules in
priority order:

1. An open fence (```` ``` ```` ... ```` ``` ````) always wins first and is
   consumed verbatim — a `|a|b|`-looking line or a `#` comment *inside* a
   code sample is never reinterpreted, because the fence scanner never
   hands those lines to the table/header detectors.
2. Outside a fence, `#`, `|`, and ```` ``` ```` are mutually exclusive
   leading tokens, so header/table-start/fence-start detection never
   genuinely competes for the same line.
3. The only real ambiguity is chart-vs-plain-fence: a ```` ```text ````
   fence only becomes `content_type="chart"` when it's immediately
   followed (at most one blank line) by a paragraph wholly wrapped in
   `*...*`/`_..._` emphasis; otherwise it's `code`/`configuration` per
   its language tag.
4. Attachment tagging never creates a block boundary of its own — it's a
   property layered onto whichever prose sub-chunk(s) literally contain
   the link, computed after normal prose splitting.
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

        k = j
        if k < n and lines[k].strip() == "":
            k += 1
        caption_lines: list[str] = []
        m = k
        while m < n and lines[m].strip() != "":
            caption_lines.append(lines[m])
            m += 1
        caption_text = "\n".join(caption_lines).strip()

        if lang and lang.lower() == "text" and caption_lines and _is_emphasis_wrapped(caption_text):
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
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if PurePosixPath(target).suffix.lower() not in _ATTACHMENT_EXTENSIONS:
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
