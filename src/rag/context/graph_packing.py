"""Graph-path-aware formatting: keep a multi-hop fact readable as one unit.

A graph path already arrives as *one* `GraphFact`/`SearchResult` (see
`rag.graph.fusion`): `GraphRetriever`/`fusion.py` never surface one
`SearchResult` per hop, so the "don't split a path into meaningless
fragments to save tokens" requirement is already satisfied structurally,
one layer below this module. What this module adds is (1) a best-effort
parse of the rendered fact text back into structured `GraphHop`s, purely
for richer display/metadata (never required for correctness), and (2) a
hop-aware shrink strategy for `rag.context.compression` that drops whole
hops from the chain's tail rather than truncating mid-sentence, which
would otherwise produce a nonsensical partial path like "Alpha API ->
Alpha Data" with no indication two more hops existed.
"""

from __future__ import annotations

import re

from rag.context.types import ContextItem, GraphHop
from rag.graph.schema import RELATIONSHIP_TYPES

#: Matches `rag.graph.retrieval._search_dependency_path`'s rendering:
#: "Dependency path: A -> B -> C" (no relation labels between hops).
_DEPENDENCY_PATH_RE = re.compile(r"^Dependency path:\s*(.+)$")

#: Matches a single-relationship fact's rendering: "Subject RELATION_TYPE
#: Object". Built from the closed `rag.graph.schema.RELATIONSHIP_TYPES`
#: vocabulary (longest names first, so "DEPENDS_ON" is tried before any
#: shorter type that happened to be a prefix of it) rather than a bare
#: "looks uppercase" pattern: several real entity names in this corpus
#: (e.g. "Alpha API") are themselves all-uppercase acronyms, which a
#: generic `[A-Z_]+` pattern would misparse as the relation itself.
_RELATIONSHIP_FACT_RE = re.compile(
    r"^(.+?)\s+(" + "|".join(sorted(RELATIONSHIP_TYPES, key=len, reverse=True)) + r")\s+(.+)$"
)


def parse_graph_fact_text(text: str) -> list[GraphHop]:
    """Best-effort structured parse of a rendered `GraphFact.text` string.

    Never raises; returns `[]` for any text that doesn't match a known
    rendering shape (e.g. an entity-only fact with no relation at all).
    This is purely a display/metadata enrichment, never load-bearing:
    `ContextItem.content` always carries the original rendered text
    regardless of whether parsing succeeds.

    Parameters
    ----------
    text : str
        A graph-origin `SearchResult.chunk.content` value.

    Returns
    -------
    list[GraphHop]
        One hop per consecutive node pair for a dependency-path
        rendering (relation left as the literal `"->"` since hop-level
        relationship types aren't preserved through to the rendered
        text); exactly one hop for a single-relationship rendering;
        `[]` otherwise.
    """
    dep_match = _DEPENDENCY_PATH_RE.match(text)
    if dep_match:
        nodes = [n.strip() for n in dep_match.group(1).split("->")]
        nodes = [n for n in nodes if n]
        return [
            GraphHop(subject=nodes[i], relation="->", object=nodes[i + 1])
            for i in range(len(nodes) - 1)
        ]
    rel_match = _RELATIONSHIP_FACT_RE.match(text)
    if rel_match:
        subject, relation, obj = rel_match.groups()
        return [GraphHop(subject=subject.strip(), relation=relation, object=obj.strip())]
    return []


def shrink_graph_path(item: ContextItem, target_tokens: int, token_counter_fn) -> str:
    """Shrink a graph-path item by dropping whole trailing hops, never mid-hop text.

    Falls back to returning `item.content` unchanged when the item has
    fewer than 2 parsed hops (nothing coherent to drop), or when no
    reduction gets under `target_tokens` (the single remaining hop is
    returned as-is; `rag.context.compression` still measures and may
    then need to drop the item outright at the packing stage).

    Parameters
    ----------
    item : ContextItem
        A `source_type == "graph_path"` item. `item.graph_path` must
        already be populated (see `parse_graph_fact_text`); an item with
        no parsed hops is returned unchanged.
    target_tokens : int
        The token budget to shrink toward.
    token_counter_fn : Callable[[str], int]
        Token-counting function (typically `TokenCounter.count`).

    Returns
    -------
    str
        The shrunk (or original) content text. Always a prefix of the
        logical hop chain, e.g. "A -> B -> C" shrinks to "A -> B", never
        "A -> B -> C" with the last node's name chopped mid-word.
    """
    hops = item.graph_path
    if len(hops) < 2:
        return item.content
    for keep in range(len(hops), 0, -1):
        kept_hops = hops[:keep]
        if kept_hops[0].relation == "->":
            nodes = [kept_hops[0].subject] + [h.object for h in kept_hops]
            candidate = "Dependency path: " + " -> ".join(nodes)
        else:
            candidate = "; ".join(f"{h.subject} {h.relation} {h.object}" for h in kept_hops)
        if token_counter_fn(candidate) <= target_tokens or keep == 1:
            return candidate
    return item.content
