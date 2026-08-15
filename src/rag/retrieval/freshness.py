"""Deterministic resolution of which document version was effective as of a given date.

Determines which members of a document-version family (linked via
`supersedes_source`) should be excluded from retrieval, from declared
metadata alone rather than leaving version selection to the LLM.
Generalizes to any chain depth via transitive-closure grouping.

Notes
-----
Limitations, by design rather than oversight:

- Families form only via `supersedes_source` links; topically related
  documents with no such link are never grouped.
- If no family member is marked `status="active"`, nothing in that
  family is excluded, rather than guessing a version by date.
- A family member with no `effective_from` is never excluded by
  `as_of`-based resolution.
"""

from __future__ import annotations

from datetime import date

from rag.path_matching import source_matches_relevant
from rag.schemas import DocumentVersionInfo


def _links_to(candidate: DocumentVersionInfo, target: DocumentVersionInfo) -> bool:
    """Whether `candidate.supersedes_source` refers to `target` by path suffix."""
    supersedes_source = candidate.supersedes_source
    if supersedes_source is None:
        return False
    return source_matches_relevant(target.source, supersedes_source)


def _build_families(versions: list[DocumentVersionInfo]) -> list[list[DocumentVersionInfo]]:
    """Group `versions` into connected `supersedes_source` families via union-find.

    Documents with no such link are excluded from the result.
    """
    parent: dict[str, str] = {v.document_id: v.document_id for v in versions}

    def find(doc_id: str) -> str:
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for v in versions:
        for other in versions:
            if v.document_id != other.document_id and _links_to(v, other):
                union(v.document_id, other.document_id)

    groups: dict[str, list[DocumentVersionInfo]] = {}
    for v in versions:
        groups.setdefault(find(v.document_id), []).append(v)
    return [group for group in groups.values() if len(group) > 1]


def _excluded_for_current(family: list[DocumentVersionInfo]) -> set[str]:
    """Exclude every non-active member of `family`, unless none is active."""
    active_ids = {v.document_id for v in family if (v.status or "").strip().lower() == "active"}
    if not active_ids:
        return set()
    return {v.document_id for v in family} - active_ids


def _excluded_for_as_of(family: list[DocumentVersionInfo], as_of: date) -> set[str]:
    """Keep only the family member effective on `as_of`; exclude the rest.

    Members with no `effective_from` are never excluded.
    """
    dated = [(v, v.effective_from) for v in family if v.effective_from is not None]
    if not dated:
        return set()
    eligible = [(v, effective_from) for v, effective_from in dated if effective_from <= as_of]
    if not eligible:
        return {v.document_id for v, _ in dated}
    winner, _ = max(eligible, key=lambda pair: pair[1])
    return {v.document_id for v, _ in dated if v.document_id != winner.document_id}


def resolve_excluded_document_ids(
    versions: list[DocumentVersionInfo],
    as_of: date | None,
    include_superseded: bool,
) -> set[str]:
    """Compute the set of `document_id`s a freshness policy excludes from retrieval.

    Parameters
    ----------
    versions : list[DocumentVersionInfo]
        Every document's governance metadata for one dataset (see
        `VectorStore.list_document_versions`).
    as_of : date | None
        `None` for "current" (prefer `status=active`); an explicit date
        resolves each family to the version effective then.
    include_superseded : bool
        When `as_of is None`, `True` disables freshness filtering
        entirely. Ignored when `as_of` is set.

    Returns
    -------
    set[str]
        `document_id`s that should be excluded from this query's retrieval.
        Empty when there are no version families, or none are excluded.
    """
    if as_of is None and include_superseded:
        return set()

    excluded: set[str] = set()
    for family in _build_families(versions):
        if as_of is not None:
            excluded |= _excluded_for_as_of(family, as_of)
        else:
            excluded |= _excluded_for_current(family)
    return excluded
