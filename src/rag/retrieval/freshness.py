"""Deterministic resolution of which document version was effective as of a given date.

Determines which members of a document-version family (linked via
`supersedes_source`) should be excluded from retrieval, from declared
metadata alone rather than leaving version selection to the LLM.
Generalizes to any chain depth via transitive-closure grouping.

Notes
-----
Rules for current queries (`as_of` unset):

- One `status="active"` member: every other member is excluded.
- Several active members: an active member superseded, directly or
  transitively, by another active member is excluded. If several
  unsuperseded active members remain, the one with the latest
  `effective_from` wins (a missing date sorts first), ties broken by the
  lexicographically greatest `document_id`. Both cases are logged.
- No active member: nothing is excluded, rather than guessing.

Historical queries (`as_of` set) ignore `status`: the member with the latest
`effective_from <= as_of` wins, members without `effective_from` are never
excluded, and ties break by `document_id`. Families form only via
`supersedes_source` links and never span two tenants or two datasets.
"""

from __future__ import annotations

import logging
from datetime import date

from rag.path_matching import path_parts, source_matches_relevant
from rag.schemas import DocumentVersionInfo

logger = logging.getLogger(__name__)

_Link = tuple[str, str]


def _links_to(candidate: DocumentVersionInfo, target: DocumentVersionInfo) -> bool:
    """Whether `candidate.supersedes_source` refers to `target` by path suffix.

    Two documents with different, non-null `tenant_id`s, or different,
    non-null `dataset_id`s, never link, even when their paths share a
    matching suffix (e.g. two tenants each authoring their own
    `pricing.md`). A `None` on either side (pre-governance content, or
    versions fetched for a single dataset) never blocks the match.
    """
    supersedes_source = candidate.supersedes_source
    if supersedes_source is None:
        return False
    for mine, theirs in (
        (candidate.tenant_id, target.tenant_id),
        (candidate.dataset_id, target.dataset_id),
    ):
        if mine is not None and theirs is not None and mine != theirs:
            return False
    return source_matches_relevant(target.source, supersedes_source)


def _supersession_links(versions: list[DocumentVersionInfo]) -> list[_Link]:
    """Return every `(newer_id, older_id)` pair where newer's `supersedes_source` names older.

    Uses an index of every trailing path suffix of each `source`, so the
    cost is linear in the number of path segments rather than quadratic in
    the number of documents. The link rule is exactly `_links_to`.
    """
    by_suffix: dict[tuple[str, ...], list[DocumentVersionInfo]] = {}
    for v in versions:
        parts = path_parts(v.source)
        for start in range(len(parts)):
            by_suffix.setdefault(parts[start:], []).append(v)
    links: list[_Link] = []
    for v in versions:
        if v.supersedes_source is None:
            continue
        for other in by_suffix.get(path_parts(v.supersedes_source), ()):
            if v.document_id != other.document_id and _links_to(v, other):
                links.append((v.document_id, other.document_id))
    return links


def _build_families(
    versions: list[DocumentVersionInfo], links: list[_Link] | None = None
) -> list[list[DocumentVersionInfo]]:
    """Group `versions` into connected `supersedes_source` families via union-find.

    Documents with no such link are excluded from the result.
    """
    parent: dict[str, str] = {v.document_id: v.document_id for v in versions}

    def find(doc_id: str) -> str:
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    for newer, older in links if links is not None else _supersession_links(versions):
        root_a, root_b = find(newer), find(older)
        if root_a != root_b:
            parent[root_a] = root_b

    groups: dict[str, list[DocumentVersionInfo]] = {}
    for v in versions:
        groups.setdefault(find(v.document_id), []).append(v)
    return [group for group in groups.values() if len(group) > 1]


def _older_by_newer(links: list[_Link]) -> dict[str, set[str]]:
    """Adjacency map from each document to the documents it directly supersedes."""
    adjacency: dict[str, set[str]] = {}
    for newer, older in links:
        adjacency.setdefault(newer, set()).add(older)
    return adjacency


def _is_active(version: DocumentVersionInfo) -> bool:
    """Whether `status` reads `active`, ignoring case and surrounding whitespace."""
    return (version.status or "").strip().lower() == "active"


def _excluded_for_current(
    family: list[DocumentVersionInfo],
    older_by_newer: dict[str, set[str]],
    issues: dict[str, list[str]],
) -> set[str]:
    """Keep exactly one active member of `family` (see the module notes); exclude the rest.

    Returns an empty set when no member is active. The kept `document_id` of
    a family with several active members is appended to
    `issues["multiple_active"]`, and also to `issues["conflicting"]` when the
    `supersedes` links did not decide between them.
    """
    active = {v.document_id: v for v in family if _is_active(v)}
    if not active:
        return set()
    winner_id = next(iter(active))
    if len(active) > 1:
        superseded: set[str] = set()
        stack = [older for doc_id in active for older in older_by_newer.get(doc_id, ())]
        while stack:
            doc_id = stack.pop()
            if doc_id not in superseded:
                superseded.add(doc_id)
                stack.extend(older_by_newer.get(doc_id, ()))
        heads = [v for doc_id, v in active.items() if doc_id not in superseded]
        candidates = heads or list(active.values())
        winner_id = max(
            candidates, key=lambda v: (v.effective_from or date.min, v.document_id)
        ).document_id
        issues.setdefault("multiple_active", []).append(winner_id)
        if len(candidates) > 1:
            issues.setdefault("conflicting", []).append(winner_id)
    return {v.document_id for v in family} - {winner_id}


def _log_issues(issues: dict[str, list[str]]) -> None:
    """Emit at most one warning per issue kind for one resolution, with a sample of ids."""
    for kind, event in (
        ("multiple_active", "freshness_multiple_active_family_members"),
        ("conflicting", "freshness_conflicting_active_versions"),
    ):
        kept = issues.get(kind)
        if kept:
            logger.warning(
                event, extra={"family_count": len(kept), "kept_document_ids": sorted(kept)[:5]}
            )


def _excluded_for_as_of(family: list[DocumentVersionInfo], as_of: date) -> set[str]:
    """Keep only the family member effective on `as_of`; exclude the rest.

    Members with no `effective_from` are never excluded. A tie on
    `effective_from` is broken deterministically by `document_id`
    (lexicographically greatest wins) rather than by list/row order.
    """
    dated = [(v, v.effective_from) for v in family if v.effective_from is not None]
    if not dated:
        return set()
    eligible = [(v, effective_from) for v, effective_from in dated if effective_from <= as_of]
    if not eligible:
        return {v.document_id for v, _ in dated}
    winner, _ = max(eligible, key=lambda pair: (pair[1], pair[0].document_id))
    return {v.document_id for v, _ in dated if v.document_id != winner.document_id}


def _match_requested_source(
    source: str, versions: list[DocumentVersionInfo], tenant_id: str | None
) -> DocumentVersionInfo | None:
    """Return the one document `source` names, or `None` when none or several match.

    An exact path match wins over suffix matches. With several candidates and
    a known `tenant_id`, other tenants' documents are dropped first; anything
    still ambiguous is logged and not resolved, never picked by row order.
    """
    requested = path_parts(source)
    candidates = [v for v in versions if path_parts(v.source) == requested] or [
        v for v in versions if source_matches_relevant(v.source, source)
    ]
    if len(candidates) > 1 and tenant_id is not None:
        scoped = [v for v in candidates if v.tenant_id in (tenant_id, None)]
        candidates = scoped or candidates
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        logger.warning(
            "freshness_ambiguous_source_reference",
            extra={"source": source, "match_count": len(candidates)},
        )
    return None


def resolve_current_document_source(
    source: str,
    versions: list[DocumentVersionInfo],
    as_of: date | None = None,
    include_superseded: bool = False,
    tenant_id: str | None = None,
) -> str:
    """Resolve any family member's `source` to its currently-effective member's `source`.

    A version family's members each have their own, distinct `source`
    path (e.g. `policy-v1.md` superseded by `policy-v2.md`). Unlike
    `resolve_excluded_document_ids`, which only reports which
    `document_id`s to exclude, this answers "if a caller names an
    arbitrary (possibly superseded) family member, what path should be
    fetched instead." Used by the `get_latest_document` agent tool so
    that naming an old version's path still resolves to current content.

    Parameters
    ----------
    source : str
        Any family member's `source` path, matched by suffix (see
        `rag.path_matching.source_matches_relevant`).
    versions : list[DocumentVersionInfo]
        Every document's governance metadata for one dataset (see
        `VectorStore.list_document_versions`).
    as_of : date | None, optional
        Same semantics as `resolve_excluded_document_ids`.
    include_superseded : bool, optional
        Same semantics as `resolve_excluded_document_ids`.
    tenant_id : str | None, optional
        The caller's tenant, used only to disambiguate a `source` that
        suffix-matches documents of several tenants.

    Returns
    -------
    str
        The resolved, currently-effective member's `source`. Returns
        `source` unchanged if it matches no known document or several
        (see `_match_requested_source`), if the matched document belongs to
        no `supersedes_source` family, or if resolution can't determine a
        winner (e.g. no family member is `status="active"` and `as_of` is
        `None`). Never guesses.
    """
    matched = _match_requested_source(source, versions, tenant_id)
    if matched is None:
        return source

    links = _supersession_links(versions)
    families = _build_families(versions, links)
    family = next(
        (f for f in families if any(v.document_id == matched.document_id for v in f)), None
    )
    if family is None:
        return matched.source

    if as_of is not None:
        excluded_ids = _excluded_for_as_of(family, as_of)
    elif include_superseded:
        excluded_ids = set()
    else:
        issues: dict[str, list[str]] = {}
        excluded_ids = _excluded_for_current(family, _older_by_newer(links), issues)
        _log_issues(issues)

    candidates = [v for v in family if v.document_id not in excluded_ids]
    if not candidates:
        return matched.source
    if len(candidates) == 1:
        return candidates[0].source
    # More than one candidate remains (rare - e.g. multiple family members
    # share effective_from); prefer the most recently effective one rather
    # than guessing arbitrarily, breaking any remaining tie deterministically
    # by document_id (lexicographically greatest wins), same rule as
    # `_excluded_for_as_of`.
    dated = [v for v in candidates if v.effective_from is not None]
    if not dated:
        return matched.source
    return max(dated, key=lambda v: (v.effective_from, v.document_id)).source


def _excluded_for_history(family: list[DocumentVersionInfo], as_of: date | None) -> set[str]:
    """Keep every member already effective on `as_of` (all, without a date); exclude later ones."""
    if as_of is None:
        return set()
    return {
        v.document_id for v in family if v.effective_from is not None and v.effective_from > as_of
    }


def resolve_excluded_document_ids(
    versions: list[DocumentVersionInfo],
    as_of: date | None,
    include_superseded: bool,
    version_scope: str = "current",
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
    version_scope : str, optional
        `"current"` (the default) applies the rules above. `"history"`
        keeps every family member effective on or before `as_of` (every
        member when `as_of` is `None`) and excludes only versions that
        took effect later, for questions comparing versions.

    Returns
    -------
    set[str]
        `document_id`s that should be excluded from this query's retrieval.
        Empty when there are no version families, or none are excluded.
    """
    if version_scope == "history":
        families = _build_families(versions)
        return set().union(*(_excluded_for_history(f, as_of) for f in families))
    if as_of is None and include_superseded:
        return set()

    links = _supersession_links(versions)
    older_by_newer = _older_by_newer(links)
    excluded: set[str] = set()
    issues: dict[str, list[str]] = {}
    for family in _build_families(versions, links):
        if as_of is not None:
            excluded |= _excluded_for_as_of(family, as_of)
        else:
            excluded |= _excluded_for_current(family, older_by_newer, issues)
    _log_issues(issues)
    return excluded
