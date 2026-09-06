"""Unit tests for `RequestIDMiddleware`'s request-id assignment.

Uses `fastapi.testclient.TestClient` against the real `rag.api.main.app`
and the always-available `GET /` route (no DI overrides needed -- it only
depends on `get_config`, which loads the real default config with no I/O).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.main import app

client = TestClient(app)


def test_request_id_is_server_generated_even_with_no_header() -> None:
    """A request with no `x-request-id` header still gets a valid id back."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["x-request-id"]


def test_caller_supplied_request_id_header_is_ignored() -> None:
    """An inbound `x-request-id` header is never echoed back or trusted.

    This value also becomes `QueryResponse.request_id`/
    `AgentQueryResponse.request_id` -- the feedback loop's
    `(tenant_id, caller_key, request_id)` dedup/update key
    (`api/routers/feedback.py`). Trusting a caller-supplied header would
    let a caller force request-id collisions across unrelated answers and
    overwrite another submission's feedback row.
    """
    forged = "attacker-chosen-request-id"

    response = client.get("/", headers={"x-request-id": forged})

    assert response.headers["x-request-id"] != forged


def test_two_requests_with_the_same_forged_header_get_different_ids() -> None:
    """Repeating the same forged header across requests never collides server-side."""
    forged = "same-value-every-time"

    first = client.get("/", headers={"x-request-id": forged})
    second = client.get("/", headers={"x-request-id": forged})

    assert first.headers["x-request-id"] != second.headers["x-request-id"]
