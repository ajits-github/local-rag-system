from __future__ import annotations

import logging
from datetime import date

from rag.retrieval.freshness import resolve_current_document_source, resolve_excluded_document_ids
from rag.schemas import DocumentVersionInfo


def _v(doc_id, source, status=None, effective_from=None, supersedes_source=None, tenant_id=None):
    return DocumentVersionInfo(
        document_id=doc_id,
        source=source,
        status=status,
        document_version=None,
        effective_from=effective_from,
        supersedes_source=supersedes_source,
        tenant_id=tenant_id,
    )


def test_no_families_when_nothing_links_via_supersedes_source():
    """Documents with no supersedes_source link at all are never excluded."""
    versions = [_v("d1", "a.md", status="active"), _v("d2", "b.md", status="active")]
    assert resolve_excluded_document_ids(versions, as_of=None, include_superseded=False) == set()


def test_current_mode_excludes_non_active_family_members():
    """as_of=None excludes every non-active member of a version family."""
    versions = [
        _v("d1", "policy-v1.md", status="superseded"),
        _v("d2", "policy-v2.md", status="active", supersedes_source="policy-v1.md"),
    ]
    excluded = resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)
    assert excluded == {"d1"}


def test_current_mode_excludes_nothing_when_no_member_is_active():
    """If no family member is marked active, nothing is excluded (documented limitation)."""
    versions = [
        _v("d1", "policy-v1.md", status="draft"),
        _v("d2", "policy-v2.md", status="draft", supersedes_source="policy-v1.md"),
    ]
    assert resolve_excluded_document_ids(versions, as_of=None, include_superseded=False) == set()


def test_as_of_resolves_deterministically_to_effective_version():
    """An explicit as_of picks the version whose effective_from <= as_of, latest first."""
    versions = [
        _v("d1", "policy-v1.md", effective_from=date(2025, 1, 1)),
        _v(
            "d2",
            "policy-v2.md",
            effective_from=date(2026, 5, 15),
            supersedes_source="policy-v1.md",
        ),
    ]
    # A date after v1 but before v2 resolves to v1. v2 excluded.
    excluded = resolve_excluded_document_ids(
        versions, as_of=date(2026, 3, 15), include_superseded=False
    )
    assert excluded == {"d2"}
    # A date after v2's effective_from resolves to v2. v1 excluded.
    excluded = resolve_excluded_document_ids(
        versions, as_of=date(2026, 8, 14), include_superseded=False
    )
    assert excluded == {"d1"}


def test_as_of_before_every_version_excludes_all_dated_members():
    """A date before every family member's effective_from excludes the whole (dated) family."""
    versions = [
        _v("d1", "policy-v1.md", effective_from=date(2025, 1, 1)),
        _v(
            "d2",
            "policy-v2.md",
            effective_from=date(2026, 5, 15),
            supersedes_source="policy-v1.md",
        ),
    ]
    excluded = resolve_excluded_document_ids(
        versions, as_of=date(2024, 1, 1), include_superseded=False
    )
    assert excluded == {"d1", "d2"}


def test_as_of_ignores_undated_family_members():
    """A family member with no effective_from is never excluded. can't be placed in time."""
    versions = [
        _v("d1", "policy-v1.md", effective_from=date(2025, 1, 1)),
        _v("d2", "policy-v2.md", effective_from=None, supersedes_source="policy-v1.md"),
    ]
    excluded = resolve_excluded_document_ids(
        versions, as_of=date(2026, 1, 1), include_superseded=False
    )
    assert "d2" not in excluded
    assert excluded == set()  # d1 is both dated and the winner, so nothing is excluded either


def test_include_superseded_disables_freshness_filtering_entirely():
    """include_superseded=True with as_of=None returns no exclusions at all."""
    versions = [
        _v("d1", "policy-v1.md", status="superseded"),
        _v("d2", "policy-v2.md", status="active", supersedes_source="policy-v1.md"),
    ]
    assert resolve_excluded_document_ids(versions, as_of=None, include_superseded=True) == set()


def test_three_version_chain_resolves_correctly_at_any_depth():
    """A v1 -> v2 -> v3 chain (not just the 2-version case) resolves deterministically."""
    versions = [
        _v("d1", "policy-v1.md", effective_from=date(2024, 1, 1)),
        _v(
            "d2",
            "policy-v2.md",
            effective_from=date(2025, 1, 1),
            supersedes_source="policy-v1.md",
        ),
        _v(
            "d3",
            "policy-v3.md",
            effective_from=date(2026, 1, 1),
            supersedes_source="policy-v2.md",
        ),
    ]
    excluded = resolve_excluded_document_ids(
        versions, as_of=date(2025, 6, 1), include_superseded=False
    )
    assert excluded == {"d1", "d3"}  # only d2 (effective 2025-01-01) survives


def test_resolve_current_document_source_redirects_an_old_version_path():
    """Naming a superseded family member's source resolves to the currently-active one's.

    This is the exact case `get_latest_document` (rag.agent.tools) needs:
    an agent (or a gold question) may name an old version's path, and
    the tool must still return current content, not the stale document.
    """
    versions = [
        _v("d1", "policy-v1.md", status="superseded"),
        _v("d2", "policy-v2.md", status="active", supersedes_source="policy-v1.md"),
    ]
    assert resolve_current_document_source("policy-v1.md", versions) == "policy-v2.md"
    # Asking for the already-current path resolves to itself.
    assert resolve_current_document_source("policy-v2.md", versions) == "policy-v2.md"


def test_resolve_current_document_source_three_version_chain():
    """A source path from any point in a v1->v2->v3 chain resolves to the current member."""
    versions = [
        _v("d1", "policy-v1.md", status="superseded"),
        _v("d2", "policy-v2.md", status="superseded", supersedes_source="policy-v1.md"),
        _v("d3", "policy-v3.md", status="active", supersedes_source="policy-v2.md"),
    ]
    assert resolve_current_document_source("policy-v1.md", versions) == "policy-v3.md"
    assert resolve_current_document_source("policy-v2.md", versions) == "policy-v3.md"


def test_resolve_current_document_source_as_of_resolves_to_dated_winner():
    """An explicit as_of resolves an old-version source to whichever member was effective then."""
    versions = [
        _v("d1", "policy-v1.md", effective_from=date(2025, 1, 1)),
        _v(
            "d2",
            "policy-v2.md",
            effective_from=date(2026, 5, 15),
            supersedes_source="policy-v1.md",
        ),
    ]
    assert (
        resolve_current_document_source("policy-v1.md", versions, as_of=date(2026, 3, 15))
        == "policy-v1.md"
    )
    assert (
        resolve_current_document_source("policy-v1.md", versions, as_of=date(2026, 8, 14))
        == "policy-v2.md"
    )


def test_resolve_current_document_source_unknown_source_returned_unchanged():
    """A source matching no known document is returned unchanged. never guesses."""
    versions = [_v("d1", "policy-v1.md", status="active")]
    assert resolve_current_document_source("unrelated.md", versions) == "unrelated.md"


def test_resolve_current_document_source_no_family_returned_unchanged():
    """A document with no supersedes_source link at all resolves to itself."""
    versions = [_v("d1", "standalone.md", status="active")]
    assert resolve_current_document_source("standalone.md", versions) == "standalone.md"


def test_resolve_current_document_source_no_active_member_falls_back_to_requested():
    """No active/dated-resolvable family member. the requested source is returned unchanged."""
    versions = [
        _v("d1", "policy-v1.md", status="draft"),
        _v("d2", "policy-v2.md", status="draft", supersedes_source="policy-v1.md"),
    ]
    assert resolve_current_document_source("policy-v1.md", versions) == "policy-v1.md"


def test_different_tenants_with_matching_filenames_never_form_one_family():
    """Two documents from different tenants sharing a same-tail filename never union.

    Within one dataset_id, tenant_alpha's own pricing.md and tenant_beta's
    own pricing.md could otherwise suffix-match into one version family
    via a coincidental (or malicious) supersedes_source value, wrongly
    excluding an unrelated tenant's current document from "current"
    results. Document-level ACL is unaffected either way (this is a
    freshness-resolution concern, not an authorization bypass) -- the
    fix is that neither document ends up excluded here at all.
    """
    versions = [
        _v("d1", "pricing.md", status="active", tenant_id="tenant_alpha"),
        _v(
            "d2",
            "pricing.md",
            status="active",
            supersedes_source="pricing.md",
            tenant_id="tenant_beta",
        ),
    ]
    excluded = resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)
    assert excluded == set()
    # Neither document redirects into the other tenant's version either.
    assert resolve_current_document_source("pricing.md", versions) in {"pricing.md"}


def test_same_tenant_matching_filenames_still_form_a_family():
    """A shared tenant_id doesn't block a genuine same-tenant version family."""
    versions = [
        _v("d1", "pricing.md", status="superseded", tenant_id="tenant_alpha"),
        _v(
            "d2",
            "pricing-v2.md",
            status="active",
            supersedes_source="pricing.md",
            tenant_id="tenant_alpha",
        ),
    ]
    excluded = resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)
    assert excluded == {"d1"}


def test_null_tenant_id_still_links_normally():
    """A None tenant_id on either side never blocks a family link (pre-governance content)."""
    versions = [
        _v("d1", "policy-v1.md", status="superseded", tenant_id=None),
        _v(
            "d2",
            "policy-v2.md",
            status="active",
            supersedes_source="policy-v1.md",
            tenant_id="tenant_alpha",
        ),
    ]
    excluded = resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)
    assert excluded == {"d1"}


def test_as_of_tie_on_effective_from_breaks_deterministically_by_document_id():
    """Two family members tied on effective_from resolve to the same winner regardless of order.

    document_id "d2" > "d1" lexicographically, so d2 must win either way.
    """
    tied_date = date(2026, 1, 1)
    versions_a = [
        _v("d1", "policy-a.md", effective_from=tied_date),
        _v("d2", "policy-b.md", effective_from=tied_date, supersedes_source="policy-a.md"),
    ]
    versions_b = list(reversed(versions_a))

    excluded_a = resolve_excluded_document_ids(
        versions_a, as_of=tied_date, include_superseded=False
    )
    excluded_b = resolve_excluded_document_ids(
        versions_b, as_of=tied_date, include_superseded=False
    )

    assert excluded_a == excluded_b == {"d1"}


def test_multiple_simultaneously_active_members_logs_a_warning(caplog):
    """Two family members both status=active logs a data-integrity warning, doesn't raise."""
    versions = [
        _v("d1", "policy-v1.md", status="active"),
        _v("d2", "policy-v2.md", status="active", supersedes_source="policy-v1.md"),
    ]
    with caplog.at_level(logging.WARNING, logger="rag.retrieval.freshness"):
        excluded = resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)

    # Both stay retrievable -- the function never guesses which one to exclude.
    assert excluded == set()
    assert any(
        "freshness_multiple_active_family_members" in record.getMessage()
        for record in caplog.records
    )


def test_single_active_member_logs_no_warning(caplog):
    """The ordinary, correct case (exactly one active member) never logs the integrity warning."""
    versions = [
        _v("d1", "policy-v1.md", status="superseded"),
        _v("d2", "policy-v2.md", status="active", supersedes_source="policy-v1.md"),
    ]
    with caplog.at_level(logging.WARNING, logger="rag.retrieval.freshness"):
        resolve_excluded_document_ids(versions, as_of=None, include_superseded=False)

    assert not any(
        "freshness_multiple_active_family_members" in record.getMessage()
        for record in caplog.records
    )
