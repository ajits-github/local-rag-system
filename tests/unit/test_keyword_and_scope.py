"""Keyword-match filtering (RET-012) and dataset scoping of requests without a dataset_id."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from rag.api.request_auth import enforce_dataset_scope
from rag.config import load_config
from rag.retrieval.scope import DatasetScopeError, resolve_dataset_id, scope_filters
from rag.vectorstore.pgvector import _tokenize, rank_keyword_matches


def _rank(docs: list[str], query: str) -> list[int]:
    return [
        i
        for i, _ in rank_keyword_matches(
            [_tokenize(d) for d in docs], _tokenize(query), [f"c{i}" for i in range(len(docs))]
        )
    ]


def test_rows_sharing_no_query_term_get_no_rank():
    """A zero-score row never appears, so it cannot earn an RRF contribution."""
    docs = ["alpha retention policy", "bravo runbook steps", "charlie network diagram"]
    assert _rank(docs, "retention") == [0]
    assert _rank(docs, "quokka") == []


def test_matches_only_on_corpus_common_terms_get_no_rank():
    """'the' appears in every row; only the row with the specific term ranks."""
    docs = [
        "the retention period is thirty days",
        "the backup runs nightly",
        "the network uses vlans",
        "the deploy uses blue green",
    ]
    assert _rank(docs, "what is the retention period") == [0]


def test_all_common_query_falls_back_to_positive_scores():
    """When every query term is common, rows with a positive score still rank."""
    docs = [
        "the alpha",
        "the bravo",
        "the charlie",
        "the delta",
        "echo foxtrot",
        "golf hotel",
        "india juliet",
    ]
    assert sorted(_rank(docs, "the")) == [0, 1, 2, 3]


def test_exact_match_in_a_tiny_corpus_still_ranks():
    """A term in every row gets a negative BM25 idf; containing it still earns a rank."""
    assert sorted(_rank(["retention policy", "retention runbook"], "retention")) == [0, 1]
    assert _rank(["retention policy"], "retention") == [0]


def test_ties_break_by_chunk_id():
    """Equal scores rank by chunk_id ascending, regardless of row order."""
    docs = ["alpha", "alpha", "bravo", "charlie", "delta"]
    ranked = rank_keyword_matches(
        [_tokenize(d) for d in docs], ["alpha"], ["z", "a", "m", "n", "o"]
    )
    assert [i for i, _ in ranked] == [1, 0]


def _config(default=None, policy="all_datasets"):
    config = load_config().model_copy(deep=True)
    config.retrieval.dataset_scope.default_dataset_id = default
    config.retrieval.dataset_scope.missing_dataset_policy = policy
    return config


def test_named_dataset_always_wins():
    """A request's own dataset_id is never replaced."""
    assert resolve_dataset_id("ds1", _config(default="other", policy="reject")) == "ds1"


def test_single_dataset_deployment_fills_in_the_default():
    """With a default, a request without dataset_id is scoped to it."""
    config = _config(default="techfusion")
    assert scope_filters(None, config) == {"dataset_id": "techfusion"}
    assert scope_filters({"category": "hr"}, config) == {
        "category": "hr",
        "dataset_id": "techfusion",
    }


def test_multi_dataset_all_datasets_policy_leaves_filters_unscoped():
    """`all_datasets` keeps the request unscoped (freshness then resolves per dataset)."""
    assert scope_filters({"category": "hr"}, _config()) == {"category": "hr"}
    assert scope_filters(None, _config()) is None


def test_multi_dataset_reject_policy_refuses_and_api_returns_422():
    """`reject` raises in the shared resolver and maps to HTTP 422 at the API."""
    config = _config(policy="reject")
    with pytest.raises(DatasetScopeError):
        resolve_dataset_id(None, config)
    with pytest.raises(HTTPException) as exc:
        enforce_dataset_scope({"category": "hr"}, config)
    assert exc.value.status_code == 422
    assert enforce_dataset_scope({"dataset_id": "ds1"}, config) == {"dataset_id": "ds1"}
