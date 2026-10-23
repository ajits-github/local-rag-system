"""Idempotent Neo4j DDL: uniqueness constraints and indexes for every entity label.

Statements are generated from `rag.graph.schema.ENTITY_TYPES`, so the label
list is not duplicated. Every statement uses `IF NOT EXISTS` and is safe to
re-run.
"""

from __future__ import annotations

from rag.graph.schema import ENTITY_TYPES


def build_constraint_statements() -> list[str]:
    """Build one `canonical_id` uniqueness constraint per entity label.

    Neo4j constraints are per label, so there is one statement per label.

    Returns
    -------
    list[str]
        One `CREATE CONSTRAINT IF NOT EXISTS ...` statement per label.
    """
    return [
        f"CREATE CONSTRAINT entity_{label.lower()}_canonical_id IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.canonical_id IS UNIQUE"
        for label in sorted(ENTITY_TYPES)
    ]


def build_index_statements() -> list[str]:
    """Build `dataset_id`/`tenant_id` range indexes per label and one full-text name index.

    The full-text index is created but no query template uses it yet. Entity
    resolution uses `CONTAINS` on a label scan.

    Returns
    -------
    list[str]
        One or more `CREATE INDEX IF NOT EXISTS ...`/`CREATE FULLTEXT
        INDEX IF NOT EXISTS ...` statements.
    """
    statements = []
    for label in sorted(ENTITY_TYPES):
        statements.append(
            f"CREATE INDEX entity_{label.lower()}_dataset_id IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.dataset_id)"
        )
        statements.append(
            f"CREATE INDEX entity_{label.lower()}_tenant_id IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.tenant_id)"
        )
    label_list = "|".join(sorted(ENTITY_TYPES))
    statements.append(
        f"CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
        f"FOR (n:{label_list}) ON EACH [n.name]"
    )
    return statements


def build_all_statements() -> list[str]:
    """Return every constraint and index statement, constraints first.

    Constraints are created before indexes since a uniqueness constraint
    implicitly creates its own backing index; ordering avoids Neo4j ever
    building a redundant index it will immediately supersede.

    Returns
    -------
    list[str]
        The full ordered list of idempotent DDL statements.
    """
    return build_constraint_statements() + build_index_statements()
