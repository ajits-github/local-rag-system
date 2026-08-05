from __future__ import annotations

from rag.vectorstore.base import ALLOWED_FILTER_FIELDS


def test_dataset_id_is_an_allowed_filter_field():
    # Regression guard: dataset_id must stay filterable, or the isolation
    # mechanism (eval/run_eval.py's mandatory dataset_id filter) silently
    # breaks with a ValueError at query time.
    assert "dataset_id" in ALLOWED_FILTER_FIELDS


def test_category_is_still_an_allowed_filter_field():
    assert "category" in ALLOWED_FILTER_FIELDS
