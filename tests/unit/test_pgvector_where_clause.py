from __future__ import annotations

import pytest

from rag.retrieval.authorization import AuthorizationContext
from rag.vectorstore.pgvector import (
    _build_where_clause,
    _combine_where_clauses,
    _tokenize,
    build_authorization_where_clause,
)


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


def test_build_authorization_where_clause_none_auth_returns_empty():
    """None auth produces no fragment and no params -- fully unrestricted."""
    assert build_authorization_where_clause(None, ["techfusion_support"]) == ("", [])


def test_build_authorization_where_clause_binds_tenant_and_roles():
    """A non-None auth binds tenant_id, the caller's cross-tenant-support-role subset, and roles."""
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])
    sql, params = build_authorization_where_clause(auth, ["techfusion_support"])
    assert "tenant_id IS NULL" in sql
    assert "tenant_id = %s" in sql
    assert "allowed_roles && %s::text[]" in sql
    # params: [tenant_id, caller_support_roles, caller_roles]
    assert params[0] == "tenant_alpha"
    assert params[1] == []  # tenant_alpha_operator is not a configured support role
    assert params[2] == ["tenant_alpha_operator"]


def test_build_authorization_where_clause_isolates_caller_support_roles():
    """Only roles present in BOTH auth.roles and cross_tenant_support_roles count as a grant."""
    auth = AuthorizationContext(tenant_id="tenant_beta", roles=["techfusion_support", "employee"])
    _sql, params = build_authorization_where_clause(auth, ["techfusion_support"])
    assert params[1] == ["techfusion_support"]  # the intersection, not the full role list
    assert params[2] == ["techfusion_support", "employee"]  # full caller roles for role_ok check


def test_build_authorization_where_clause_appends_excluded_document_ids():
    """A non-empty resolved_excluded_document_ids adds a third exclusion clause."""
    auth = AuthorizationContext(
        tenant_id="tenant_alpha", roles=[], resolved_excluded_document_ids=["doc-1", "doc-2"]
    )
    sql, params = build_authorization_where_clause(auth, [])
    assert "NOT (document_id::text = ANY(%s))" in sql
    assert params[-1] == ["doc-1", "doc-2"]


def test_build_authorization_where_clause_omits_exclusion_clause_when_empty():
    """No resolved_excluded_document_ids means no third exclusion clause/param."""
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=[])
    sql, params = build_authorization_where_clause(auth, [])
    assert "document_id::text = ANY" not in sql
    assert len(params) == 3


def test_combine_where_clauses_both_empty():
    """No filters, no auth -> no WHERE clause at all."""
    assert _combine_where_clauses("", "") == ""


def test_combine_where_clauses_filters_only():
    """Filters alone pass through unchanged."""
    assert _combine_where_clauses("WHERE dataset_id = %s", "") == "WHERE dataset_id = %s"


def test_combine_where_clauses_auth_only_gets_where_prefix():
    """Auth alone (no caller filters) still produces a valid WHERE clause."""
    assert _combine_where_clauses("", "tenant_id IS NULL") == "WHERE (tenant_id IS NULL)"


def test_build_authorization_where_clause_appends_trust_level_requirement():
    """require_trust_level adds a NULL-permissive trust_level clause."""
    auth = AuthorizationContext(require_trust_level="authoritative")
    sql, params = build_authorization_where_clause(auth, [])
    assert "trust_level IS NULL OR trust_level = %s" in sql
    assert params[-1] == "authoritative"


def test_build_authorization_where_clause_omits_trust_clause_when_unset():
    """A None require_trust_level (the default) adds no trust_level clause at all."""
    auth = AuthorizationContext()
    sql, _params = build_authorization_where_clause(auth, [])
    assert "trust_level" not in sql


def test_combine_where_clauses_both_present_are_anded():
    """Filters and auth combine via AND, auth wrapped in its own parens."""
    combined = _combine_where_clauses("WHERE dataset_id = %s", "tenant_id IS NULL")
    assert combined == "WHERE dataset_id = %s AND (tenant_id IS NULL)"
