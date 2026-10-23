"""Shared fixtures for `tests/unit`.

The autouse fixture below exists because a container-only `RAG_CONFIG_PATH`
value, once loaded into a local shell's environment, quietly broke the
full unit suite hours after the test that set it had finished.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_host_invalid_config_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete `RAG_CONFIG_PATH` before every unit test, unless the test sets it itself.

    `rag.config.load_config()` calls `load_dotenv(override=False)`, so a
    developer's own `.env` (which may point `RAG_CONFIG_PATH` at a
    container-only path like `/app/config/production.yaml` for real
    Docker runs) gets baked into `os.environ` the first time any test
    (or even module import during collection) triggers a config load,
    and stays there for the rest of the process. Combined with any test
    that clears `rag.api.deps.get_config`'s `lru_cache` (e.g.
    `test_api_config_override.py`), the next unrelated test to build a
    fresh `get_config()` would fail trying to load a path that only
    exists inside a container.

    Unit tests must never depend on a host machine's own `.env`; a test
    that wants to exercise `RAG_CONFIG_PATH` sets it explicitly via its
    own `monkeypatch.setenv(...)`, which layers on top of (and is
    reverted together with) this fixture's `delenv` at the end of that
    same test, so this is a true default, never a test-order hazard.
    """
    monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)


# --- Ontology milestone fixtures (test_ontology_*.py) ------------------
# No real Postgres/Ollama/ML-model calls, same as every other tests/unit
# fixture: these load small Turtle files into an in-memory rdflib.Graph.
# `rag.ontology`/`rdflib` are imported INSIDE each fixture, not at this
# conftest's module scope, so collecting the full tests/unit suite never
# requires the optional `ontology` extra (`pip install -e .[ontology]`)
# to be installed. Only running an ontology test does, the same lazy-
# import discipline `rag.eval.ragas_scorer` already uses for `ragas`.


@pytest.fixture
def ontology_valid_store():
    """Ontology + both valid instance-data files + interoperability/resolution layer."""
    from rag.ontology import paths as ontology_paths
    from rag.ontology.store import load_store

    return load_store(ontology_paths.ALL_VALID_FILES)


@pytest.fixture
def ontology_only_store():
    """Just the ontology T-box, no instance data."""
    from rag.ontology import paths as ontology_paths
    from rag.ontology.store import load_store

    return load_store([ontology_paths.ONTOLOGY_V1])


@pytest.fixture
def ontology_shapes_store():
    """Load the SHACL shapes graph."""
    from rag.ontology import paths as ontology_paths
    from rag.ontology.store import load_store

    return load_store([ontology_paths.SHAPES])


@pytest.fixture
def ontology_invalid_store():
    """Ontology + customer-domain data + the deliberately invalid fixture."""
    from rag.ontology import paths as ontology_paths
    from rag.ontology.store import load_store

    return load_store(
        [
            ontology_paths.ONTOLOGY_V1,
            ontology_paths.INSTANCES_CUSTOMER_DOMAIN,
            ontology_paths.INSTANCES_INVALID,
        ]
    )


@pytest.fixture
def ontology_reasoning_fixtures_store():
    """Ontology + the OWL-RL reasoning fixtures (functional-property, disjointness)."""
    from rag.ontology import paths as ontology_paths
    from rag.ontology.store import load_store

    return load_store([ontology_paths.ONTOLOGY_V1, ontology_paths.REASONING_FIXTURES])
