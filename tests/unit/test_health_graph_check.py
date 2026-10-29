"""`GET /health` probes the graph through the Neo4j client alone (OBS-001).

`/health` used to depend on `get_graphrag_pipeline`, which builds the
retrieval pipeline and loads the embedding model, so a cold first health
check paid seconds of model-load cost just to ping Neo4j.
"""

from __future__ import annotations

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from rag.api import deps
from rag.api.main import app
from rag.api.routers.health import router as health_router
from rag.config import load_config


class _Healthy:
    def health_check(self) -> bool:
        return True


class _Unreachable:
    def health_check(self) -> bool:
        return False


def _health_dependency_calls() -> set:
    route = next(r for r in health_router.routes if isinstance(r, APIRoute) and r.path == "/health")

    def walk(dependant):
        for sub in dependant.dependencies:
            yield sub.call
            yield from walk(sub)

    return set(walk(route.dependant))


def test_health_never_depends_on_the_retrieval_or_graphrag_pipeline():
    """No dependency of `/health`, direct or transitive, builds a heavy pipeline singleton."""
    calls = _health_dependency_calls()

    assert deps.get_graph_client in calls
    assert deps.get_graphrag_pipeline not in calls
    assert deps.get_retrieval_pipeline not in calls
    assert deps.get_embedder not in calls


def _get_health(graph_enabled: bool, graph_client):
    config = load_config().model_copy(deep=True)
    config.graph.enabled = graph_enabled
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_vectorstore] = lambda: _Healthy()
    app.dependency_overrides[deps.get_llm] = lambda: _Healthy()
    app.dependency_overrides[deps.get_graph_client] = lambda: graph_client
    try:
        return TestClient(app).get("/health").json()
    finally:
        for dep in (deps.get_config, deps.get_vectorstore, deps.get_llm, deps.get_graph_client):
            app.dependency_overrides.pop(dep, None)


def test_disabled_graph_reads_as_disabled_not_degraded():
    """With no graph client, the endpoint stays ok and labels the graph disabled."""
    body = _get_health(graph_enabled=False, graph_client=None)

    assert body["status"] == "ok"
    assert body["dependencies"]["graph"] == "disabled"


def test_unreachable_neo4j_degrades_health():
    """An enabled graph whose client cannot reach Neo4j reports degraded."""
    body = _get_health(graph_enabled=True, graph_client=_Unreachable())

    assert body["status"] == "degraded"
    assert body["dependencies"]["graph"] == "unreachable"
