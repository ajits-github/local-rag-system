from __future__ import annotations

import pytest

from rag.vectorstore.pgvector import _build_where_clause, _tokenize


def test_build_where_clause_none_filters_returns_empty():
    """None filters produce no WHERE clause and no params."""
    assert _build_where_clause(None) == ("", [])


def test_build_where_clause_empty_dict_returns_empty():
    """An empty filters dict produces no WHERE clause and no params."""
    assert _build_where_clause({}) == ("", [])


def test_build_where_clause_single_filter():
    """One filter produces a single-condition WHERE clause with its value as a param."""
    where_sql, params = _build_where_clause({"dataset_id": "techfusion"})
    assert where_sql == "WHERE dataset_id = %s"
    assert params == ["techfusion"]


def test_build_where_clause_multiple_filters_joined_with_and():
    """Multiple filters are AND-joined, params ordered to match the %s placeholders."""
    where_sql, params = _build_where_clause({"dataset_id": "techfusion", "category": "security"})
    assert where_sql == "WHERE dataset_id = %s AND category = %s"
    assert params == ["techfusion", "security"]


def test_build_where_clause_rejects_disallowed_key():
    """A filter key outside ALLOWED_FILTER_FIELDS raises ValueError -- the SQL-injection guard."""
    with pytest.raises(ValueError, match="not allowed"):
        _build_where_clause({"chunk_id": "doc-1_0"})


def test_tokenize_strips_punctuation_attached_to_words():
    """A JSON-style token like '"maximum_wait_minutes":' tokenizes to match a plain word query.

    Regression test for a real bug: a plain whitespace split left
    punctuation attached (quotes, colons) so a query for
    'maximum_wait_minutes' never matched JSON content containing it.
    """
    tokens = _tokenize('"maximum_wait_minutes": 240,')
    assert "maximum_wait_minutes" in tokens
    query_tokens = _tokenize("maximum_wait_minutes")
    assert query_tokens == ["maximum_wait_minutes"]
    assert set(query_tokens) & set(tokens)


def test_tokenize_lowercases_and_splits_on_whitespace():
    """Tokenization lowercases and splits ordinary prose on word boundaries."""
    assert _tokenize("The Quick Brown Fox.") == ["the", "quick", "brown", "fox"]


def test_tokenize_preserves_underscores_within_identifiers():
    """An underscore-joined identifier stays a single token, not split into parts."""
    assert _tokenize("retry_transient(fn)") == ["retry_transient", "fn"]
