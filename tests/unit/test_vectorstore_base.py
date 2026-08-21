from __future__ import annotations

from rag.vectorstore.base import ALLOWED_FILTER_FIELDS


def test_dataset_id_is_an_allowed_filter_field():
    """dataset_id must stay filterable, or dataset isolation breaks silently."""
    # Regression guard: dataset_id must stay filterable, or the isolation
    # mechanism (eval/run_eval.py's mandatory dataset_id filter) silently
    # breaks with a ValueError at query time.
    assert "dataset_id" in ALLOWED_FILTER_FIELDS


def test_category_is_still_an_allowed_filter_field():
    """Category remains a supported metadata filter field."""
    assert "category" in ALLOWED_FILTER_FIELDS


def test_governance_fields_are_allowed_filter_fields():
    """tenant_id/classification/status/trust_level are exact-match-filterable convenience fields.

    Distinct from AuthorizationContext enforcement, which is never
    caller-controlled. These are the same "caller may narrow, never
    broaden" convenience filters category/content_type already are.
    """
    for field in ("tenant_id", "classification", "status", "trust_level", "doc_source_type"):
        assert field in ALLOWED_FILTER_FIELDS
