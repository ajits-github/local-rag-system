"""File-path constants for the ontology layer's bundled Turtle/SPARQL/Cypher files.

Resolved from `rag.config.REPO_ROOT` (the same anchor every other
`AppConfig`-adjacent path in this codebase uses), not from this source
file's own location, so these constants stay correct regardless of
whether `rag` was installed editable or copied into `site-packages`
(see `CLAUDE.md`'s "Containerized runtime" section for why that
distinction matters elsewhere in this codebase).
"""

from __future__ import annotations

from rag.config import REPO_ROOT

ONTOLOGY_DATA_ROOT = REPO_ROOT / "data" / "ontology"

SCHEMA_DIR = ONTOLOGY_DATA_ROOT / "schema"
SHAPES_DIR = ONTOLOGY_DATA_ROOT / "shapes"
INSTANCES_DIR = ONTOLOGY_DATA_ROOT / "instances"
QUERIES_DIR = ONTOLOGY_DATA_ROOT / "queries"
SPARQL_DIR = QUERIES_DIR / "sparql"
CYPHER_DIR = QUERIES_DIR / "cypher"

ONTOLOGY_V1 = SCHEMA_DIR / "techfusion_ontology_v1.ttl"
ONTOLOGY_V1_1 = SCHEMA_DIR / "techfusion_ontology_v1_1.ttl"
SHAPES = SHAPES_DIR / "techfusion_shapes.ttl"

INSTANCES_GRAPH_DEMO = INSTANCES_DIR / "instances_graph_demo.ttl"
INSTANCES_CUSTOMER_DOMAIN = INSTANCES_DIR / "instances_customer_domain.ttl"
INSTANCES_INVALID = INSTANCES_DIR / "instances_invalid.ttl"
REASONING_FIXTURES = INSTANCES_DIR / "reasoning_fixtures.ttl"
INTEROPERABILITY_MAPPINGS = INSTANCES_DIR / "interoperability_mappings.ttl"
CANONICAL_ENTITY_RESOLUTION = INSTANCES_DIR / "canonical_entity_resolution.ttl"

ONTOLOGY_GOLD = REPO_ROOT / "data" / "eval" / "ontology_gold.jsonl"

# The "whole valid world": ontology + both valid instance data files +
# the interoperability/entity-resolution layer. Used by most demos/tests.
# Deliberately excludes INSTANCES_INVALID and REASONING_FIXTURES, which
# are loaded only by the specific demo/test that needs them.
ALL_VALID_FILES = [
    ONTOLOGY_V1,
    INSTANCES_GRAPH_DEMO,
    INSTANCES_CUSTOMER_DOMAIN,
    INTEROPERABILITY_MAPPINGS,
    CANONICAL_ENTITY_RESOLUTION,
]
