"""Version-family rules: conflicting active versions, dataset boundaries, and historical queries."""

from __future__ import annotations

import logging
from datetime import date

from rag.retrieval.freshness import resolve_current_document_source, resolve_excluded_document_ids
from rag.schemas import DocumentVersionInfo


def _v(
    doc_id,
    source,
    status=None,
    effective_from=None,
    supersedes_source=None,
    tenant_id=None,
    dataset_id=None,
):
    return DocumentVersionInfo(
        document_id=doc_id,
        source=source,
        status=status,
        effective_from=effective_from,
        supersedes_source=supersedes_source,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
    )


def _current(versions):
    return resolve_excluded_document_ids(versions, None, False)


def test_new_version_published_while_old_still_active_supersedes_it(caplog):
    """v2 links to v1 and both say active: the link decides, v1 is excluded, and it is logged."""
    versions = [
        _v("d1", "p-v1.md", status="active"),
        _v("d2", "p-v2.md", status="active", supersedes_source="p-v1.md"),
    ]
    with caplog.at_level(logging.WARNING, logger="rag.retrieval.freshness"):
        assert _current(versions) == {"d1"}
    assert any("freshness_multiple_active_family_members" in r.getMessage() for r in caplog.records)
    assert resolve_current_document_source("p-v1.md", versions) == "p-v2.md"


def test_active_member_superseded_transitively_is_excluded():
    """v3 (active) -> v2 (superseded) -> v1 (still active): only v3 stays."""
    versions = [
        _v("d1", "p-v1.md", status="active"),
        _v("d2", "p-v2.md", status="superseded", supersedes_source="p-v1.md"),
        _v("d3", "p-v3.md", status="active", supersedes_source="p-v2.md"),
    ]
    assert _current(versions) == {"d1", "d2"}


def test_branching_active_heads_keep_the_latest_effective_one(caplog):
    """Two active versions both supersede v1: latest `effective_from` wins, logged as a conflict."""
    versions = [
        _v("d1", "p-v1.md", status="superseded", effective_from=date(2025, 1, 1)),
        _v(
            "d2",
            "p-v2a.md",
            status="active",
            effective_from=date(2026, 1, 1),
            supersedes_source="p-v1.md",
        ),
        _v(
            "d3",
            "p-v2b.md",
            status="active",
            effective_from=date(2026, 3, 1),
            supersedes_source="p-v1.md",
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="rag.retrieval.freshness"):
        assert _current(versions) == {"d1", "d2"}
    assert any("freshness_conflicting_active_versions" in r.getMessage() for r in caplog.records)


def test_branching_heads_tie_breaks_by_document_id_regardless_of_order():
    """Equal (or missing) dates: lexicographically greatest document_id wins, in any input order."""
    versions = [
        _v("d1", "p-v1.md", status="superseded"),
        _v("da", "p-v2a.md", status="active", supersedes_source="p-v1.md"),
        _v("db", "p-v2b.md", status="active", supersedes_source="p-v1.md"),
    ]
    assert _current(versions) == {"d1", "da"}
    assert _current(list(reversed(versions))) == {"d1", "da"}


def test_dated_head_beats_undated_head():
    """A head without `effective_from` loses to a dated one."""
    versions = [
        _v("d1", "p-v1.md", status="superseded"),
        _v("dz", "p-v2a.md", status="active", supersedes_source="p-v1.md"),
        _v(
            "da",
            "p-v2b.md",
            status="active",
            effective_from=date(2026, 1, 1),
            supersedes_source="p-v1.md",
        ),
    ]
    assert _current(versions) == {"d1", "dz"}


def test_cycle_of_active_versions_still_resolves_to_one():
    """A supersedes cycle has no head; the tie-break still keeps exactly one member."""
    versions = [
        _v("d1", "p-v1.md", status="active", supersedes_source="p-v2.md"),
        _v("d2", "p-v2.md", status="active", supersedes_source="p-v1.md"),
    ]
    assert _current(versions) == {"d1"}


def test_no_active_member_is_still_left_alone():
    """Unchanged documented rule: no active member, nothing excluded."""
    versions = [
        _v("d1", "p-v1.md", status="draft"),
        _v("d2", "p-v2.md", status="draft", supersedes_source="p-v1.md"),
    ]
    assert _current(versions) == set()


def test_historical_query_ignores_status():
    """`as_of` picks by `effective_from` even when the older version says active."""
    versions = [
        _v("d1", "p-v1.md", status="active", effective_from=date(2025, 1, 1)),
        _v(
            "d2",
            "p-v2.md",
            status="active",
            effective_from=date(2026, 1, 1),
            supersedes_source="p-v1.md",
        ),
    ]
    assert resolve_excluded_document_ids(versions, date(2025, 6, 1), False) == {"d2"}
    assert resolve_excluded_document_ids(versions, date(2026, 6, 1), False) == {"d1"}


def test_versions_never_link_across_datasets():
    """The same path in two datasets: dataset B's v2 must not exclude dataset A's document."""
    versions = [
        _v("a1", "kb/p-v1.md", status="active", dataset_id="A"),
        _v("b1", "kb/p-v1.md", status="superseded", dataset_id="B"),
        _v("b2", "kb/p-v2.md", status="active", supersedes_source="p-v1.md", dataset_id="B"),
    ]
    assert _current(versions) == {"b1"}


def test_unscoped_resolution_over_several_datasets_resolves_each_family():
    """Versions from two datasets in one list resolve independently."""
    versions = [
        _v("a1", "p-v1.md", status="superseded", dataset_id="A"),
        _v("a2", "p-v2.md", status="active", supersedes_source="p-v1.md", dataset_id="A"),
        _v("b1", "q-v1.md", status="superseded", dataset_id="B"),
        _v("b2", "q-v2.md", status="active", supersedes_source="q-v1.md", dataset_id="B"),
    ]
    assert _current(versions) == {"a1", "b1"}


# ---------------------------------------------------------------------------
# version_scope="history" (questions comparing versions)
# ---------------------------------------------------------------------------

_CHAIN = [
    _v("d1", "p-v1.md", status="superseded", effective_from=date(2025, 1, 1)),
    _v(
        "d2",
        "p-v2.md",
        status="active",
        effective_from=date(2026, 1, 1),
        supersedes_source="p-v1.md",
    ),
    _v(
        "d3",
        "p-v3.md",
        status="draft",
        effective_from=date(2027, 1, 1),
        supersedes_source="p-v2.md",
    ),
]


def test_history_without_a_date_keeps_every_version():
    """No `as_of`: every family member is retrievable for a comparison question."""
    assert resolve_excluded_document_ids(_CHAIN, None, False, "history") == set()


def test_history_with_a_date_excludes_only_later_versions():
    """`as_of` mid-2026: v1 and v2 had taken effect, v3 had not."""
    assert resolve_excluded_document_ids(_CHAIN, date(2026, 6, 1), False, "history") == {"d3"}


def test_history_keeps_undated_members():
    """A member without `effective_from` cannot be placed in time, so it is never excluded."""
    versions = [*_CHAIN[:2], _v("dx", "p-vx.md", status="draft", supersedes_source="p-v2.md")]
    assert resolve_excluded_document_ids(versions, date(2025, 6, 1), False, "history") == {"d2"}


def test_history_never_links_across_tenants():
    """Scope changes which versions are kept, not how families form."""
    versions = [
        _v("a1", "a/p.md", status="superseded", tenant_id="ta", effective_from=date(2027, 1, 1)),
        _v("b2", "b/p-v2.md", status="active", tenant_id="tb", supersedes_source="p.md"),
    ]
    assert resolve_excluded_document_ids(versions, date(2026, 1, 1), False, "history") == set()
