"""Deterministic resolution of which document version was effective as of a given date.

Adjustment to the original design: rather than making every superseded
version eligible and leaving version selection to the LLM, this module
resolves -- in Python, from declared metadata alone -- exactly which
member of a document-version family (linked via `supersedes_source`, see
`schemas.DocumentVersionInfo`) should be excluded from retrieval for a given
`AuthorizationContext.as_of`/`include_superseded` policy. Generalizes to any
chain depth (v1 -> v2 -> v3 -> ...) via full transitive-closure grouping,
not just the two-version case seen in the current corpus.

**Documented limitations** (deliberate, not hidden):
- A family is only formed via `supersedes_source` path-suffix links. A
  document with no `supersedes`/no document supersedes it is never grouped,
  even if it's topically related to another document -- this module never
  guesses a family from naming similarity.
- Current-mode (`as_of=None`) resolution relies on `status="active"` being
  set correctly on exactly the current member. If **no** member of a family
  is marked active, nothing in that family is excluded (safer than guessing
  a version by date when the corpus doesn't declare one).
- Historical-mode (`as_of=<date>`) resolution relies on `effective_from`
  being set. A family member with no `effective_from` can't be placed in
  time, so it is never excluded by this logic (kept eligible) -- callers
  that need a hard guarantee here must ensure `effective_from` is always
  authored.
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
    """Group `versions` into version families via `supersedes_source` links (union-find).

    A "family" is a connected component of 2+ documents linked (in either
    direction, any chain depth) by `supersedes_source`. Documents with no
    such link at all are never returned (nothing to resolve for them).
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
    """Within one family, exclude every non-`active` member -- unless none is active."""
    active_ids = {v.document_id for v in family if (v.status or "").strip().lower() == "active"}
    if not active_ids:
        return set()
    return {v.document_id for v in family} - active_ids


def _excluded_for_as_of(family: list[DocumentVersionInfo], as_of: date) -> set[str]:
    """Within one family, keep only the member effective on `as_of`; exclude the rest.

    Members with no `effective_from` are never excluded (can't be placed in
    time -- see module docstring).
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
        `None` for "current" (prefer `status=active`); an explicit date to
        deterministically resolve each family to the version effective then.
    include_superseded : bool
        When `as_of is None`, `True` disables freshness filtering entirely
        (returns an empty set). Ignored when `as_of` is set -- an explicit
        date always resolves deterministically, per this milestone's
        "don't leave version selection to the LLM" requirement.

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
