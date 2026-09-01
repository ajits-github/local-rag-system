"""`config.mcp.enabled=False` (the shipped default) leaves `rag.api.main.app`'s routes untouched.

Imports the real app object -- built once, at module-import time, from
whatever config the test process resolves (`config/default.yaml` unless
`RAG_CONFIG_PATH` is set, matching every other test in this file's
sibling `test_root_endpoint.py`) -- and proves neither the bare mount
path nor its trailing-slash form is routed anywhere: both return a
genuine 404, indistinguishable from a path that was never a route at
all. `tests/integration/test_mcp_end_to_end.py` and
`tests/integration/test_mcp_tenant_isolation.py` prove the enabled case
(they build `build_mcp_asgi_app` directly, since flipping this
process-wide `app` singleton to `mcp.enabled=True` would require
rebuilding its real embedder/vectorstore/pipeline dependencies, which
this file deliberately keeps out of scope, matching
`test_root_endpoint.py`'s own "no dependency injection" framing).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag.api.main import app
from rag.config import load_config


def test_mcp_disabled_by_default():
    """Sanity check the assumption the rest of this file depends on."""
    assert load_config().mcp.enabled is False


def test_bare_and_trailing_slash_mcp_paths_both_404_when_disabled():
    """Neither spelling of the MCP mount path is routed when mcp.enabled=False."""
    client = TestClient(app)
    for path in ("/mcp", "/mcp/", "/mcp/anything"):
        response = client.get(path)
        assert response.status_code == 404, f"{path} was routed even though mcp.enabled=False"


def test_mcp_disabled_matches_a_route_that_never_existed():
    """A disabled MCP mount 404s exactly like a path that was never a route -- a true no-op."""
    client = TestClient(app)
    mcp_response = client.get("/mcp")
    control_response = client.get("/this-route-genuinely-does-not-exist")
    assert mcp_response.status_code == control_response.status_code == 404
