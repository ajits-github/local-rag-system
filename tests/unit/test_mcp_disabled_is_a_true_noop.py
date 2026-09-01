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
from starlette.routing import Mount

import rag.api.main as main_module
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


def test_no_mcp_app_is_ever_constructed_when_disabled():
    """The MCP server, and its Streamable HTTP session manager/lifespan, is never built.

    `rag/api/main.py`'s module-level `_mcp_app = build_mcp_asgi_app(...) if
    config.mcp.enabled else None` short-circuits: when disabled,
    `build_mcp_asgi_app` (and everything under it -- `build_mcp_server`,
    tool registration, the SDK's session manager) is never called at
    all, not merely built and then left unmounted. This is a stronger
    property than route-table absence alone: no MCP background task
    group is ever created for `_lifespan` to enter, since there is
    nothing to enter.
    """
    assert main_module._mcp_app is None


def test_no_mount_route_for_the_mcp_path_exists_in_the_route_table_when_disabled():
    """The app's own route table has no `Mount` registered at the MCP mount path at all.

    A second, independent check from the HTTP-behavior tests above:
    inspects `app.routes` directly rather than inferring absence from a
    404 response (which could, in principle, come from some other
    handler).
    """
    mount_paths = {route.path for route in app.routes if isinstance(route, Mount)}
    assert "/mcp" not in mount_paths
