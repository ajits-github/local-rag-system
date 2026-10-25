"""Oversized-evidence and tool-output compression.

Handles section 10 (an individual item larger than its available
budget) and section 11 (MCP/tool-output compression, most of which
already happens at the source; see the module docstring note below)
with explicit, structural strategies, never an arbitrary character-
offset cut:

- `"prose"` (and any other/unknown `content_type`): sentence-boundary
  trimming. `"extractive"` additionally scores sentences by query-term
  overlap and keeps the highest-scoring ones *in their original order*
  (never reordered: reordering sentences out of their original sequence
  risks changing their meaning). `"sentence_trim"` just keeps leading
  sentences up to budget, a simpler fallback.
- `"table"`/`"code"`/`"configuration"`: line-boundary trimming, since a
  sentence regex would cut a table row or a code statement in half. For
  a table, the header line (first 1-2 lines) is always kept so a
  partial table never loses its column meaning.
- `"graph_path"`: whole-hop trimming via `rag.context.graph_packing.
  shrink_graph_path`, never mid-hop text.

Compression never reintroduces redacted data: it only ever *selects a
subset of already-existing text* (sentences, lines, or hops already
present in `item.content`, which itself already passed through
`RetrievalPipeline.sanitize_evidence` before `ContextItem` was built).
There is no decoding, no reconstruction, and no LLM call in this
module, so a `[REDACTED:SENSITIVE_FIELD]` marker already in the text
can be kept or dropped by a selection, but never un-redacted or
reassembled from fragments.

Note on tool-output compression (section 11): the schema-validation ->
sanitization -> relevant-field-selection -> bounded-representation
pipeline the design describes already exists at the MCP tool-output
*source*, not here: `rag.agent.mcp_client._business_result_to_search_result`
re-validates a raw MCP result against its Pydantic schema and renders
only a handful of named fields into a short prose string (see that
module), so a verbose raw JSON payload never reaches `ContextItem` in
the first place. This module's job for `mcp_tool` items is the same
generic oversized-text handling every other bucket gets, nothing MCP-specific.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from rag.context.graph_packing import shrink_graph_path
from rag.context.types import ContextItem

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

_LINE_BASED_CONTENT_TYPES = frozenset({"table", "code", "configuration"})


def _split_sentences(text: str) -> list[str]:
    """Split `text` on sentence-ending punctuation; never drops any text."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p for p in parts if p]


def _query_terms(query: str) -> set[str]:
    """Lowercased alphanumeric query words, for extractive sentence scoring."""
    return {w.lower() for w in _WORD_RE.findall(query)}


def _score_sentence(sentence: str, terms: set[str]) -> int:
    """Count of query terms appearing in `sentence` (case-insensitive)."""
    if not terms:
        return 0
    words = {w.lower() for w in _WORD_RE.findall(sentence)}
    return len(words & terms)


def _compress_prose(
    text: str,
    *,
    target_tokens: int,
    count_tokens: Callable[[str], int],
    query: str,
    strategy: str,
    max_sentences: int,
) -> str:
    """Compress prose text toward `target_tokens` via sentence selection."""
    sentences = _split_sentences(text)
    if not sentences:
        return text
    if strategy == "extractive":
        terms = _query_terms(query)
        scored = sorted(
            range(len(sentences)), key=lambda i: _score_sentence(sentences[i], terms), reverse=True
        )
        keep_indices: set[int] = set()
        running_tokens = 0
        for idx in scored:
            if len(keep_indices) >= max_sentences:
                break
            sentence_tokens = count_tokens(sentences[idx])
            if keep_indices and running_tokens + sentence_tokens > target_tokens:
                continue
            keep_indices.add(idx)
            running_tokens += sentence_tokens
        if not keep_indices:
            keep_indices = {scored[0]}
        ordered = sorted(keep_indices)
        return " ".join(sentences[i] for i in ordered)

    # "sentence_trim": keep leading sentences until the budget is spent.
    kept: list[str] = []
    running_tokens = 0
    for sentence in sentences[:max_sentences]:
        sentence_tokens = count_tokens(sentence)
        if kept and running_tokens + sentence_tokens > target_tokens:
            break
        kept.append(sentence)
        running_tokens += sentence_tokens
    if not kept:
        kept = [sentences[0]]
    return " ".join(kept)


def _compress_lines(
    text: str, *, target_tokens: int, count_tokens: Callable[[str], int], max_lines: int
) -> str:
    """Compress line-structured text (table/code/configuration) by keeping whole lines.

    Always keeps the first line (a table header or an opening code-fence/
    key line) so a partial table/config block never loses its column/
    context meaning.
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    header = lines[0]
    kept = [header]
    running_tokens = count_tokens(header)
    for line in lines[1 : max_lines + 1]:
        line_tokens = count_tokens(line)
        if running_tokens + line_tokens > target_tokens:
            break
        kept.append(line)
        running_tokens += line_tokens
    return "\n".join(kept)


def compress_item(
    item: ContextItem,
    *,
    target_tokens: int,
    count_tokens: Callable[[str], int],
    query: str,
    strategy: str,
    max_sentences: int,
) -> ContextItem:
    """Compress one oversized `ContextItem` toward `target_tokens`.

    Returns `item` unchanged (but with `original_token_count` recorded)
    when it is already within budget. Never drops `citation`/
    `source_id`/governance metadata; only `content` (and, for a
    `graph_path` item, `graph_path` to match the shrunk text) changes.

    Parameters
    ----------
    item : ContextItem
        The item to compress; `item.token_count` must already be set.
    target_tokens : int
        The budget to shrink toward. Compression is best-effort: a
        result may still exceed `target_tokens` when even one
        sentence/line/hop doesn't fit (the caller, `rag.context.packing`,
        is responsible for the final fit/drop decision).
    count_tokens : Callable[[str], int]
        Token-counting function.
    query : str
        The current query text, for extractive sentence scoring.
    strategy : {"sentence_trim", "extractive"}
        Ignored for `table`/`code`/`configuration`/`graph_path` items,
        which always use their own structural strategy.
    max_sentences : int
        Ceiling on sentences/lines kept, independent of the token budget.

    Returns
    -------
    ContextItem
        A copy of `item` with `content`/`token_count` updated and
        `compressed=True`, `original_token_count` set, when
        compression actually ran.
    """
    if item.token_count <= target_tokens:
        return item

    original_tokens = item.token_count
    if item.source_type == "graph_path" and item.graph_path:
        new_content = shrink_graph_path(item, target_tokens, count_tokens)
        compression_strategy = "graph_hop_trim"
    elif item.content_type in _LINE_BASED_CONTENT_TYPES:
        new_content = _compress_lines(
            item.content,
            target_tokens=target_tokens,
            count_tokens=count_tokens,
            max_lines=max_sentences,
        )
        compression_strategy = "line_trim"
    else:
        new_content = _compress_prose(
            item.content,
            target_tokens=target_tokens,
            count_tokens=count_tokens,
            query=query,
            strategy=strategy,
            max_sentences=max_sentences,
        )
        compression_strategy = strategy

    new_tokens = count_tokens(new_content)
    if new_tokens >= original_tokens:
        # Compression produced no reduction (e.g. a single unsplittable
        # sentence); return the item unchanged rather than claiming a
        # compression that didn't help.
        return item
    return item.model_copy(
        update={
            "content": new_content,
            "token_count": new_tokens,
            "original_token_count": original_tokens,
            "compressed": True,
            "compression_strategy": compression_strategy,
        }
    )
