"""Shared fixtures for the whole `tests/` tree (both `unit/` and `integration/`).

A parent-level `conftest.py` so an `autouse=True` fixture here applies
automatically to test modules under either subdirectory, with no explicit
import needed.
"""

from __future__ import annotations

import copy

import pytest

from rag.mcp.business import store as business_store


@pytest.fixture(autouse=True)
def _reset_case_store():
    """Restore `rag.mcp.business.store`'s shared, in-memory `_SYNTHETIC_CASES` dict.

    `update_case_status` (the one write action against the synthetic
    business-case store) genuinely mutates process-local module state, so
    any test that dispatches it -- directly, through `run_agent()`, or
    through a real MCP `ClientSession` -- needs this restored afterward or
    a later test's seeded-state assumptions break. Previously duplicated
    verbatim across three test files
    (`tests/unit/test_mcp_business_case_actions.py`,
    `tests/integration/test_agent_mcp_client_stage2.py`,
    `tests/integration/test_specialized_tool_gold_fixtures.py`); centralized
    here since every mutating test lives under one of `tests/unit` or
    `tests/integration`, both reachable from this one parent conftest.
    """
    original = copy.deepcopy(business_store._SYNTHETIC_CASES)
    yield
    business_store._SYNTHETIC_CASES.clear()
    business_store._SYNTHETIC_CASES.update(original)
