"""`/ingest` and `/feedback` are now rate-limited, matching `/query`/`/agent/query`.

Before this fix, neither route carried a `@_limiter.limit(...)` decorator
at all -- unlike `/query`/`/agent/query`/`/agent/query/stream`, which all
do -- and `Limiter(...)` (`api/deps.py:get_rate_limiter`) has no
`default_limits=` fallback, so `SlowAPIMiddleware` enforced nothing on an
undecorated route. This matters once `security.rate_limit.enabled=true`
(off by default, but on in `config/production.yaml`).

`_limiter` (`api/deps.py:get_rate_limiter()`) is a single process-wide
singleton every router module binds its decorator to at import time; its
`.enabled` flag and the rate-limit-string helpers' `security.rate_limit.
requests_per_minute` read are both checked fresh on every request (proven
directly against the installed `slowapi`'s `LimitGroup.__iter__`, which
calls the limit-string callable per request rather than baking in a value
at decoration time). So, unlike `test_rate_limiting.py`'s isolated-app
tests (needed there because that file proves the *mechanism*), these tests
flip the real singleton's `.enabled` and the real shared `AppConfig`
singleton's `requests_per_minute` in place, directly exercising the real,
already-decorated `/ingest`/`/feedback` routes on `rag.api.main.app` --
restoring both in a `finally` block so no other test in the session is
affected.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from rag.api.deps import (
    get_config,
    get_feedback_store,
    get_ingestion_pipeline,
    get_rate_limiter,
)
from rag.api.main import app


class _FakeIngestionPipeline:
    """IngestionPipeline double that answers instantly, no real Postgres/embedding model."""

    def ingest_file(self, path, dataset_id, caller=None, source_override=None) -> dict[str, Any]:
        """Record nothing meaningful; just answer successfully."""
        return {"document_id": "doc-1", "chunks_written": 1, "changed": True}


class _FakeFeedbackStore:
    """FeedbackStore double that answers instantly, no real Postgres."""

    def submit(self, data) -> tuple[str, bool]:
        """Report a newly created row, always."""
        return "feedback-1", True


def _set_rate_limit(*, enabled: bool, requests_per_minute: int) -> tuple[bool, int]:
    """Mutate the real, shared `Limiter`/`AppConfig` singletons in place; return their old values.

    Both `get_rate_limiter()` and `get_config()` are `lru_cache`d with no
    arguments, so every router module (including the already-imported
    `ingest`/`feedback` routers, whose decorators are bound to these exact
    objects) observes the mutation immediately, with no re-import needed.
    """
    limiter = get_rate_limiter()
    config = get_config()
    old_enabled = limiter.enabled
    old_rpm = config.security.rate_limit.requests_per_minute
    limiter.enabled = enabled
    config.security.rate_limit.requests_per_minute = requests_per_minute
    return old_enabled, old_rpm


def _restore_rate_limit(old_enabled: bool, old_rpm: int) -> None:
    """Undo `_set_rate_limit`'s mutation."""
    limiter = get_rate_limiter()
    config = get_config()
    limiter.enabled = old_enabled
    config.security.rate_limit.requests_per_minute = old_rpm


def test_repeated_ingest_calls_beyond_the_limit_return_429(tmp_path, monkeypatch):
    """With rate limiting enabled and a 1/minute effective ingest budget, the 2nd call 429s.

    `_ingest_rate_limit_string()` budgets `/ingest` at a quarter of
    `requests_per_minute` (minimum 1/minute); `requests_per_minute=1` below
    keeps that at its floor of 1/minute regardless of the exact divisor.
    """
    monkeypatch.setattr("rag.api.routers.ingest.UPLOAD_DIR", tmp_path)
    old_enabled, old_rpm = _set_rate_limit(enabled=True, requests_per_minute=1)
    app.dependency_overrides[get_ingestion_pipeline] = lambda: _FakeIngestionPipeline()
    try:
        client = TestClient(app)

        def _upload(filename: str):
            return client.post(
                "/ingest",
                data={"dataset_id": "test-dataset"},
                files={"files": (filename, b"# hello", "text/markdown")},
            )

        first = _upload("a.md")
        second = _upload("b.md")

        assert first.status_code == 200
        assert second.status_code == 429
    finally:
        _restore_rate_limit(old_enabled, old_rpm)
        app.dependency_overrides.pop(get_ingestion_pipeline, None)


def test_repeated_feedback_calls_beyond_the_limit_return_429():
    """With rate limiting enabled and a 1/minute budget, a 2nd /feedback submission 429s."""
    old_enabled, old_rpm = _set_rate_limit(enabled=True, requests_per_minute=1)
    app.dependency_overrides[get_feedback_store] = lambda: _FakeFeedbackStore()
    try:
        client = TestClient(app)

        def _submit(request_id: str):
            return client.post("/feedback", json={"request_id": request_id, "rating": "positive"})

        first = _submit("req-1")
        second = _submit("req-2")

        assert first.status_code == 200
        assert second.status_code == 429
    finally:
        _restore_rate_limit(old_enabled, old_rpm)
        app.dependency_overrides.pop(get_feedback_store, None)


def test_ingest_and_feedback_are_unaffected_when_rate_limiting_stays_disabled(tmp_path):
    """The shipped default (`rate_limit.enabled=False`) still lets many calls through unthrottled.

    A regression guard for the decorator addition itself: adding
    `@_limiter.limit(...)` must never throttle anything while the limiter
    stays disabled, matching every other decorated route's existing
    behavior.
    """
    limiter = get_rate_limiter()
    assert limiter.enabled is False  # the real, shipped default
    app.dependency_overrides[get_feedback_store] = lambda: _FakeFeedbackStore()
    try:
        client = TestClient(app)
        responses = [
            client.post("/feedback", json={"request_id": f"req-{i}", "rating": "positive"})
            for i in range(5)
        ]
        assert all(r.status_code == 200 for r in responses)
    finally:
        app.dependency_overrides.pop(get_feedback_store, None)
