"""Static proof of tool isolation: each specialist module never *imports* the other's tool surface.

Mirrors the repo's own static-boundary-test idiom (`tests/unit/
test_gold_data_isolation.py`'s "static import-boundary test" for
`rag.eval`/`rag.ingestion`/`rag.retrieval`), adapted to check actual
`import`/`from ... import` statements via `ast` rather than a raw
substring search -- a substring search over the *whole* source would
also match this module's own docstring prose (which necessarily *names*
the forbidden modules to explain why they're absent), producing a false
failure. Parsing only the import statements is what makes this a genuine
structural check.

This is the concrete evidence behind README.md's "tool isolation is
structural, not a runtime permission check" claim (see `knowledge_agent.
py`/`business_agent.py`'s own module docstrings): a specialist cannot
call the other's tool not because a check forbids it at runtime, but
because the code that could do so was never written in that file.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _imported_module_names(module_filename: str) -> set[str]:
    """Return every dotted module name this file imports (`import x.y` or `from x.y import z`)."""
    tree = ast.parse((_PACKAGE_ROOT / module_filename).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _no_import_starts_with(imported: set[str], forbidden_prefix: str) -> bool:
    """Return True if no imported module name equals or is nested under `forbidden_prefix`."""
    return not any(
        name == forbidden_prefix or name.startswith(forbidden_prefix + ".") for name in imported
    )


def test_knowledge_agent_never_imports_the_business_case_store():
    """The Knowledge Agent module must never import the business case store."""
    imported = _imported_module_names("knowledge_agent.py")
    assert _no_import_starts_with(imported, "rag.mcp")
    assert _no_import_starts_with(imported, "rag.agent.mcp_client")
    assert "business_agent" not in imported


def test_business_agent_never_imports_the_retrieval_pipeline():
    """The Business Agent module must never import the retrieval pipeline."""
    imported = _imported_module_names("business_agent.py")
    assert _no_import_starts_with(imported, "rag.retrieval.pipeline")
    assert _no_import_starts_with(imported, "rag.agent.tools")
    assert "knowledge_agent" not in imported


def test_orchestrator_never_imports_a_tool_dispatch_module_directly():
    """The orchestration tier reads state written by specialists but never dispatches a tool itself.

    `rag.agent.graph` is imported for its pure text classifier
    (`_looks_like_case_mutation_request`) only -- allowed, since it is a
    regex function, not a tool call; the point of this test is that
    `orchestrator.py` never imports the modules that actually reach
    Postgres/the case store (`rag.retrieval.pipeline.RetrievalPipeline`,
    `rag.mcp.business.store`).
    """
    imported = _imported_module_names("orchestrator.py")
    assert _no_import_starts_with(imported, "rag.retrieval.pipeline")
    assert _no_import_starts_with(imported, "rag.mcp.business.store")
