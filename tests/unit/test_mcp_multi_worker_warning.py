"""`rag.api.main`'s startup warning for multi-worker deployments with MCP write actions.

`rag.mcp.business.store`'s in-memory `_SYNTHETIC_CASES` dict and every
`api/deps.py` singleton are process-local; a `--workers N > 1` deployment
would silently duplicate them per worker, breaking cross-worker
consistency for `update_case_status` with no error. Since nothing
currently prevents `--workers N > 1`, `_warn_if_multi_worker_mcp_business_actions`
logs a best-effort warning (from the `WEB_CONCURRENCY` env var convention)
rather than raising, since that convention isn't universally reliable.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from rag.api.main import _detect_worker_count, _warn_if_multi_worker_mcp_business_actions


def _config(*, business_actions_enabled: bool) -> SimpleNamespace:
    """Build a minimal object exposing only the attribute path the warning reads."""
    return SimpleNamespace(
        mcp=SimpleNamespace(business_actions=SimpleNamespace(enabled=business_actions_enabled))
    )


def test_warns_when_multi_worker_and_business_actions_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """WEB_CONCURRENCY > 1 plus mcp.business_actions.enabled=True logs the warning."""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with caplog.at_level(logging.WARNING, logger="rag.api.main"):
        _warn_if_multi_worker_mcp_business_actions(_config(business_actions_enabled=True))

    assert any(
        "mcp_business_actions_multi_worker_risk" in record.getMessage() for record in caplog.records
    )


def test_no_warning_at_a_single_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """WEB_CONCURRENCY=1 (the correct, single-worker deployment) never warns."""
    monkeypatch.setenv("WEB_CONCURRENCY", "1")
    with caplog.at_level(logging.WARNING, logger="rag.api.main"):
        _warn_if_multi_worker_mcp_business_actions(_config(business_actions_enabled=True))

    assert caplog.records == []


def test_no_warning_when_web_concurrency_is_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """An unset WEB_CONCURRENCY (worker count genuinely unknown) never warns -- no false alarm."""
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    with caplog.at_level(logging.WARNING, logger="rag.api.main"):
        _warn_if_multi_worker_mcp_business_actions(_config(business_actions_enabled=True))

    assert caplog.records == []


def test_no_warning_when_business_actions_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A multi-worker signal with business_actions.enabled=False (no mutating tool) never warns."""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with caplog.at_level(logging.WARNING, logger="rag.api.main"):
        _warn_if_multi_worker_mcp_business_actions(_config(business_actions_enabled=False))

    assert caplog.records == []


def test_detect_worker_count_parses_a_valid_integer(monkeypatch: pytest.MonkeyPatch):
    """_detect_worker_count returns the parsed int when WEB_CONCURRENCY is a valid integer."""
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    assert _detect_worker_count() == 3


def test_detect_worker_count_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch):
    """_detect_worker_count returns None, never a guessed default, when the env var is unset."""
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    assert _detect_worker_count() is None


def test_detect_worker_count_returns_none_for_a_malformed_value(monkeypatch: pytest.MonkeyPatch):
    """A non-integer WEB_CONCURRENCY value is treated as unknown, not an error."""
    monkeypatch.setenv("WEB_CONCURRENCY", "not-a-number")
    assert _detect_worker_count() is None
