"""Shared Markdown-equivalent-syntax rendering helpers for layout-aware loaders.

`PDFLoader`/`DocxLoader` both serialize extracted structure into the same
syntax `StructuredMarkdownChunker` parses (see that module's docstring);
this module holds the couple of rendering rules that are genuinely
identical between the two (table syntax, a lightweight language sniff for
otherwise-untagged code/config blocks) so neither loader duplicates them.
"""

from __future__ import annotations

import re

_JSON_START_RE = re.compile(r"^\s*[{\[]")
_XML_START_RE = re.compile(r"^\s*<")
_YAML_LINE_RE = re.compile(r"^\s*[\w][\w.\-]*:\s")


def render_table(rows: list[list[str | None]]) -> str:
    """Render `rows` (header row first, then data rows) as a Markdown pipe table.

    Parameters
    ----------
    rows : list[list[str | None]]
        Table rows, each a list of cell strings (`None` for an empty/
        merged cell, e.g. from `pdfplumber`'s `Table.extract()`); `rows[0]`
        is the header.

    Returns
    -------
    str
        Markdown pipe-table text (header, separator, data rows), or `""`
        if `rows` is empty.
    """
    clean_rows = [[cell or "" for cell in row] for row in rows if row]
    if not clean_rows:
        return ""
    header, *data_rows = clean_rows
    width = len(header)

    def render_row(row: list[str]) -> str:
        cells = list(row) + [""] * (width - len(row))
        return "| " + " | ".join(c.replace("\n", " ").strip() for c in cells[:width]) + " |"

    lines = [render_row(header), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(render_row(row) for row in data_rows)
    return "\n".join(lines)


def sniff_code_language(block_text: str) -> str | None:
    """Best-effort language guess for a monospace-styled block with no explicit tag.

    Parameters
    ----------
    block_text : str
        The block's joined text.

    Returns
    -------
    str | None
        `"json"`/`"xml"`/`"yaml"` when the shape is recognizable enough to
        tag as `content_type="configuration"` downstream, else `None`
        (renders as an untagged, plain `content_type="code"` fence). Never
        claims a language it can't reasonably infer from shape alone.
    """
    stripped = block_text.strip()
    if not stripped:
        return None
    if _JSON_START_RE.match(stripped):
        return "json"
    if _XML_START_RE.match(stripped):
        return "xml"
    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and sum(1 for line in lines if _YAML_LINE_RE.match(line)) >= max(1, len(lines) // 2):
        return "yaml"
    return None
