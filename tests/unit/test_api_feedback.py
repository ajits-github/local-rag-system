"""Unit tests for POST /feedback: validation, identity/tenant trust, dedup/update, failure handling.

Uses `fastapi.testclient.TestClient` against the real `rag.api.main.app`
with `get_config`/`get_current_identity`/`get_feedback_store` overridden,
matching `test_api_dos_limits.py`/`test_api_query_auth_boundary.py`'s
established pattern -- no real Postgres needed.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_config, get_current_identity, get_feedback_store
from rag.api.main import app
from rag.config import load_config
from rag.feedback.store import FeedbackWriteInput

_SECRET = "unit-test-only-not-a-real-secret-value"


class _RecordingFeedbackStore:
    """FeedbackStore double implementing the real dedup/update key in memory.

    Mirrors `FeedbackStore.submit`'s `(tenant_id, caller_key, request_id)`
    upsert semantics without touching Postgres, so duplicate/update
    behavior can be asserted precisely.
    """

    def __init__(self, fail: bool = False) -> None:
        """Start with no rows; `fail=True` makes every `submit()` raise."""
        self._rows: dict[tuple[str, str, str], tuple[str, FeedbackWriteInput]] = {}
        self.calls: list[FeedbackWriteInput] = []
        self.fail = fail

    def submit(self, data: FeedbackWriteInput) -> tuple[str, bool]:
        """Record the call and apply the same dedup key as the real store."""
        self.calls.append(data)
        if self.fail:
            raise RuntimeError("simulated database failure")
        key = (data.tenant_id or "", data.caller_key, data.request_id)
        if key in self._rows:
            feedback_id, _ = self._rows[key]
            self._rows[key] = (feedback_id, data)
            return feedback_id, False
        feedback_id = str(uuid.uuid4())
        self._rows[key] = (feedback_id, data)
        return feedback_id, True


def _feedback_config(**overrides: Any):
    """Load the default config with `feedback.*` overridden; security.auth stays at its default."""
    config = load_config()
    if overrides:
        feedback = config.feedback.model_copy(update=overrides)
        config = config.model_copy(update={"feedback": feedback})
    return config


def _auth_enabled_config(**feedback_overrides: Any):
    """Load config with security.auth enabled and HS256 configured for tests."""
    config = _feedback_config(**feedback_overrides)
    jwt_config = config.security.auth.jwt.model_copy(update={"secret_env_var": "JWT_HS256_SECRET"})
    auth_config = config.security.auth.model_copy(update={"enabled": True, "jwt": jwt_config})
    security = config.security.model_copy(update={"auth": auth_config})
    return config.model_copy(update={"security": security})


def _token(**claim_overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "tenant_id": "tenant_alpha",
        "roles": ["tenant_alpha_operator"],
    }
    claims.update({"iat": now, "exp": now + 3600})
    claims.update(claim_overrides)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


@pytest.fixture
def client_with(monkeypatch):
    """Build a `(client, store)` pair with `get_config`/`get_feedback_store` overridden."""
    monkeypatch.setenv("JWT_HS256_SECRET", _SECRET)

    def _build(config=None, store=None):
        config = config if config is not None else load_config()
        store = store if store is not None else _RecordingFeedbackStore()
        app.dependency_overrides[get_config] = lambda: config
        app.dependency_overrides[get_feedback_store] = lambda: store
        return TestClient(app), store

    yield _build
    app.dependency_overrides.pop(get_config, None)
    app.dependency_overrides.pop(get_feedback_store, None)
    app.dependency_overrides.pop(get_current_identity, None)


def test_positive_feedback_is_accepted_and_recorded(client_with):
    """A minimal positive-rating submission (no auth) is accepted and stored."""
    client, store = client_with()

    response = client.post("/feedback", json={"request_id": "req-1", "rating": "positive"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "created"
    assert len(store.calls) == 1
    assert store.calls[0].rating == "positive"
    assert store.calls[0].caller_key == "anonymous"


def test_negative_feedback_with_reason_and_comment_is_accepted(client_with):
    """A full negative-rating submission with a matching reason and a comment is accepted."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={
            "request_id": "req-2",
            "rating": "negative",
            "reason": "wrong_or_missing_citation",
            "comment": "cited the wrong document",
            "query": "what is the retention policy?",
            "answer": "42 days.",
            "route": "classic_rag",
            "dataset_id": "techfusion",
            "cited_source_ids": ["doc_0", "doc_1"],
        },
    )

    assert response.status_code == 200
    call = store.calls[0]
    assert call.reason == "wrong_or_missing_citation"
    assert call.comment == "cited the wrong document"
    assert call.query_text == "what is the retention policy?"
    assert call.answer_text == "42 days."
    assert call.cited_source_ids == ["doc_0", "doc_1"]


def test_missing_optional_comment_is_accepted(client_with):
    """Omitting comment/reason/query/answer entirely is a valid submission."""
    client, store = client_with()

    response = client.post("/feedback", json={"request_id": "req-3", "rating": "negative"})

    assert response.status_code == 200
    assert store.calls[0].comment is None
    assert store.calls[0].reason is None


def test_invalid_rating_returns_422(client_with):
    """A rating outside {"positive", "negative"} is rejected with 422."""
    client, store = client_with()

    response = client.post("/feedback", json={"request_id": "req-4", "rating": "meh"})

    assert response.status_code == 422
    assert store.calls == []


@pytest.mark.parametrize(
    "rating,reason",
    [("positive", "incorrect_answer"), ("negative", "accurate_and_helpful")],
)
def test_reason_mismatched_with_rating_returns_422(client_with, rating, reason):
    """A reason from the other rating's enum is rejected with 422."""
    client, store = client_with()

    response = client.post(
        "/feedback", json={"request_id": "req-5", "rating": rating, "reason": reason}
    )

    assert response.status_code == 422
    assert store.calls == []


def test_unrecognized_reason_returns_422(client_with):
    """A reason outside the fixed vocabulary entirely is rejected with 422."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={"request_id": "req-6", "rating": "negative", "reason": "not_a_real_reason"},
    )

    assert response.status_code == 422
    assert store.calls == []


def test_oversized_comment_returns_422(client_with):
    """A comment longer than feedback.max_comment_length is rejected with 422."""
    config = _feedback_config(max_comment_length=20)
    client, store = client_with(config)

    response = client.post(
        "/feedback", json={"request_id": "req-7", "rating": "positive", "comment": "x" * 21}
    )

    assert response.status_code == 422
    assert store.calls == []


def test_oversized_cited_source_ids_returns_422(client_with):
    """cited_source_ids longer than feedback.max_cited_sources is rejected with 422."""
    config = _feedback_config(max_cited_sources=2)
    client, store = client_with(config)

    response = client.post(
        "/feedback",
        json={"request_id": "req-8", "rating": "positive", "cited_source_ids": ["a", "b", "c"]},
    )

    assert response.status_code == 422
    assert store.calls == []


def test_unknown_field_returns_422(client_with):
    """extra='forbid' rejects an unrecognized field rather than silently ignoring it."""
    client, store = client_with()

    response = client.post(
        "/feedback", json={"request_id": "req-9", "rating": "positive", "not_a_real_field": "x"}
    )

    assert response.status_code == 422
    assert store.calls == []


def test_feedback_disabled_returns_404(client_with):
    """config.feedback.enabled=False 404s the route, matching the DoS-limits check pattern."""
    config = _feedback_config(enabled=False)
    client, store = client_with(config)

    response = client.post("/feedback", json={"request_id": "req-10", "rating": "positive"})

    assert response.status_code == 404
    assert store.calls == []


def test_forged_tenant_id_is_ignored_when_jwt_present(client_with):
    """A verified JWT's tenant_id wins over a conflicting body tenant_id (cross-tenant attempt)."""
    client, store = client_with(_auth_enabled_config())
    token = _token(tenant_id="tenant_alpha", roles=["tenant_alpha_operator"])

    response = client.post(
        "/feedback",
        json={"request_id": "req-11", "rating": "positive", "tenant_id": "tenant_beta"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert store.calls[0].tenant_id == "tenant_alpha"


def test_unauthenticated_request_rejected_when_auth_enabled(client_with):
    """No Authorization header at all is rejected with 401 when insecure_dev_mode is off."""
    client, store = client_with(_auth_enabled_config())

    response = client.post("/feedback", json={"request_id": "req-12", "rating": "positive"})

    assert response.status_code == 401
    assert store.calls == []


def test_no_jwt_or_auth_context_ever_reaches_the_store(client_with):
    """The FeedbackWriteInput the store receives carries no JWT/AuthorizationContext object."""
    client, store = client_with(_auth_enabled_config())
    token = _token()

    client.post(
        "/feedback",
        json={"request_id": "req-13", "rating": "positive"},
        headers={"Authorization": f"Bearer {token}"},
    )

    call = store.calls[0]
    # Every field is a plain str/list[str]/None -- never a VerifiedIdentity, claims dict, or token.
    for value in vars(call).values():
        assert value is None or isinstance(value, str | list)
    assert call.caller_key != "alice"  # pseudonymized, never the raw JWT subject


def test_duplicate_submission_by_same_caller_updates_existing_row(client_with):
    """A second submission for the same (tenant, caller, request_id) updates, not duplicates."""
    client, store = client_with()

    first = client.post("/feedback", json={"request_id": "req-14", "rating": "positive"})
    second = client.post(
        "/feedback",
        json={"request_id": "req-14", "rating": "negative", "reason": "incorrect_answer"},
    )

    assert first.json()["status"] == "created"
    assert second.json()["status"] == "updated"
    assert first.json()["feedback_id"] == second.json()["feedback_id"]
    assert len(store.calls) == 2


def test_stale_or_nonexistent_request_id_is_still_accepted_deterministically(client_with):
    """An arbitrary request_id not tied to any real prior request is still accepted.

    No server-side request log exists to validate against (see
    `api/routers/feedback.py`'s module docstring); this is a documented,
    deliberate choice, not an oversight.
    """
    client, store = client_with()

    response = client.post(
        "/feedback", json={"request_id": "this-was-never-a-real-request-id", "rating": "positive"}
    )

    assert response.status_code == 200
    assert len(store.calls) == 1


def test_database_failure_returns_503(client_with):
    """A store.submit() failure is surfaced as a 503, not a raw 500 leaking internals."""
    client, store = client_with(store=_RecordingFeedbackStore(fail=True))

    response = client.post("/feedback", json={"request_id": "req-15", "rating": "positive"})

    assert response.status_code == 503
    assert "database" not in response.json()["detail"].lower()  # no internal detail leaked


def test_feedback_disabled_never_resolves_the_feedback_store_dependency(client_with):
    """`get_feedback_store` is never invoked when feedback is disabled.

    Regression test for a real gap: `get_feedback_store` is an
    `lru_cache`d singleton that eagerly opens a Postgres connection pool
    in `FeedbackStore.__init__`. Before `_require_feedback_enabled` was
    declared ahead of it in `submit_feedback`'s signature, FastAPI would
    still resolve that dependency on every call (constructing/opening a
    real pool) even when `config.feedback.enabled=False`, so a deployment
    that intends feedback to be fully absent could still fail on an
    unreachable database instead of cleanly 404ing.
    """
    resolve_calls: list[int] = []

    def _spy_get_feedback_store() -> object:
        resolve_calls.append(1)
        return object()

    config = _feedback_config(enabled=False)
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_feedback_store] = _spy_get_feedback_store
    try:
        response = TestClient(app).post(
            "/feedback", json={"request_id": "req-16", "rating": "positive"}
        )
    finally:
        app.dependency_overrides.pop(get_config, None)
        app.dependency_overrides.pop(get_feedback_store, None)

    assert response.status_code == 404
    assert resolve_calls == []


def test_oversized_tenant_id_returns_422(client_with):
    """A tenant_id longer than the hard 200-char identifier bound is rejected with 422."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={"request_id": "req-17", "rating": "positive", "tenant_id": "x" * 201},
    )

    assert response.status_code == 422
    assert store.calls == []


def test_oversized_dataset_id_returns_422(client_with):
    """A dataset_id longer than the hard 200-char identifier bound is rejected with 422."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={"request_id": "req-18", "rating": "positive", "dataset_id": "x" * 201},
    )

    assert response.status_code == 422
    assert store.calls == []


def test_oversized_cited_source_id_item_returns_422(client_with):
    """A single overlong cited_source_ids entry is rejected even though the list is short."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={
            "request_id": "req-19",
            "rating": "positive",
            "cited_source_ids": ["x" * 201],
        },
    )

    assert response.status_code == 422
    assert store.calls == []


def test_oversized_tool_calls_item_returns_422(client_with):
    """A single overlong tool_calls entry is rejected even though the list is short."""
    client, store = client_with()

    response = client.post(
        "/feedback",
        json={"request_id": "req-20", "rating": "positive", "tool_calls": ["x" * 201]},
    )

    assert response.status_code == 422
    assert store.calls == []
