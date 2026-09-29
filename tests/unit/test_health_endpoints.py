"""Endpoint-semantics tests for /health, /livez, /readyz.

/livez must stay 200 regardless of dependency state (a Postgres/Ollama
outage is never a reason to restart an otherwise-healthy process).
/readyz must go non-2xx whenever a dependency required to serve a normal
request is down (so Kubernetes stops routing traffic to this Pod without
restarting it). /health stays 200 either way, degradation reported only
in the body -- unchanged behavior, covered here for the same-file
regression-guard convenience.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api.deps import get_llm, get_vectorstore
from rag.api.main import app


class _StubDependency:
    """A vectorstore/LLM double whose `health_check()` return value is fixed."""

    def __init__(self, healthy: bool) -> None:
        """Fix this double's reachability for the lifetime of the test."""
        self._healthy = healthy

    def health_check(self) -> bool:
        """Return the fixed reachability value, ignoring all other calls."""
        return self._healthy


@pytest.fixture
def client_with_dependencies():
    """Build a TestClient with overridable vectorstore/LLM health."""

    def _make(db_ok: bool, llm_ok: bool) -> TestClient:
        app.dependency_overrides[get_vectorstore] = lambda: _StubDependency(db_ok)
        app.dependency_overrides[get_llm] = lambda: _StubDependency(llm_ok)
        return TestClient(app)

    yield _make
    app.dependency_overrides.pop(get_vectorstore, None)
    app.dependency_overrides.pop(get_llm, None)


def test_livez_returns_200_when_all_dependencies_healthy(client_with_dependencies):
    """/livez is 200 in the ordinary healthy case."""
    client = client_with_dependencies(db_ok=True, llm_ok=True)

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_livez_returns_200_when_vectorstore_unreachable(client_with_dependencies):
    """/livez must not fail just because Postgres is temporarily unavailable."""
    client = client_with_dependencies(db_ok=False, llm_ok=True)

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_livez_returns_200_when_llm_unreachable(client_with_dependencies):
    """/livez must not fail just because Ollama is temporarily unavailable."""
    client = client_with_dependencies(db_ok=True, llm_ok=False)

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_livez_returns_200_when_both_dependencies_unreachable(client_with_dependencies):
    """/livez stays 200 even when every downstream dependency is down."""
    client = client_with_dependencies(db_ok=False, llm_ok=False)

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readyz_returns_200_when_all_dependencies_healthy(client_with_dependencies):
    """/readyz is 200 when both required dependencies are reachable."""
    client = client_with_dependencies(db_ok=True, llm_ok=True)

    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"] == {"vectorstore": "ok", "llm": "ok"}


def test_readyz_returns_non_2xx_when_vectorstore_unreachable(client_with_dependencies):
    """/readyz goes non-2xx when the vectorstore, required for retrieval, is down."""
    client = client_with_dependencies(db_ok=False, llm_ok=True)

    response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["dependencies"]["vectorstore"] == "unreachable"


def test_readyz_returns_non_2xx_when_llm_unreachable(client_with_dependencies):
    """/readyz goes non-2xx when the LLM, required for generation, is down."""
    client = client_with_dependencies(db_ok=True, llm_ok=False)

    response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["dependencies"]["llm"] == "unreachable"


def test_readyz_returns_non_2xx_when_both_dependencies_unreachable(client_with_dependencies):
    """/readyz goes non-2xx when every required dependency is down."""
    client = client_with_dependencies(db_ok=False, llm_ok=False)

    response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["dependencies"] == {"vectorstore": "unreachable", "llm": "unreachable"}


def test_health_stays_200_but_reports_degraded_when_a_dependency_is_down(
    client_with_dependencies,
):
    """/health's existing contract (always 200, degradation in the body) is unchanged."""
    client = client_with_dependencies(db_ok=False, llm_ok=True)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"] == {"vectorstore": "unreachable", "llm": "ok"}
