from __future__ import annotations

import pytest

from rag.graph.cypher_guard import CypherValidationError, validate_generated_cypher


def test_accepts_a_well_formed_query_and_appends_a_trusted_limit():
    """A valid, bounded, dataset-scoped read query is accepted and gets a fresh LIMIT."""
    cypher = "MATCH (e {dataset_id: $dataset_id})-[:DEPENDS_ON]->(o) RETURN e, o LIMIT 5"
    result = validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)
    assert "LIMIT 50" in result
    assert result.count("LIMIT") == 1  # the original LIMIT 5 was stripped, not left dangling


def test_rejects_empty_query():
    """A blank/whitespace-only query is rejected."""
    with pytest.raises(CypherValidationError):
        validate_generated_cypher("   ", max_depth=3, max_result_limit=50)


@pytest.mark.parametrize(
    "keyword",
    [
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH DELETE",
        "SET",
        "REMOVE",
        "DROP",
        "CALL",
        "LOAD CSV",
        "FOREACH",
    ],
)
def test_rejects_every_mutation_or_procedure_keyword(keyword):
    """Every write/procedure keyword the milestone's constraint list names is rejected."""
    cypher = f"MATCH (e {{dataset_id: $dataset_id}}) {keyword} (e) RETURN e"
    with pytest.raises(CypherValidationError):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_rejects_multiple_statements():
    """A second statement after a semicolon is rejected outright."""
    cypher = "MATCH (e {dataset_id: $dataset_id}) RETURN e; MATCH (x) DETACH DELETE x"
    with pytest.raises(CypherValidationError):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_rejects_a_comment():
    """A `//` comment anywhere in the query is rejected (an obfuscation vector)."""
    cypher = "MATCH (e {dataset_id: $dataset_id}) RETURN e // sneaky comment"
    with pytest.raises(CypherValidationError):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_rejects_unbounded_variable_length_pattern():
    """A bare `*` with no upper bound is always rejected, regardless of max_depth."""
    cypher = "MATCH (e {dataset_id: $dataset_id})-[*]->(o) RETURN o"
    with pytest.raises(CypherValidationError, match="unbounded"):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_rejects_variable_length_pattern_exceeding_max_depth():
    """A bounded pattern whose upper bound exceeds `max_depth` is rejected."""
    cypher = "MATCH (e {dataset_id: $dataset_id})-[*1..10]->(o) RETURN o"
    with pytest.raises(CypherValidationError, match="max_depth"):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_accepts_variable_length_pattern_within_max_depth():
    """A bounded pattern within `max_depth` is accepted."""
    cypher = "MATCH (e {dataset_id: $dataset_id})-[*1..2]->(o) RETURN o LIMIT 10"
    result = validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)
    assert "*1..2" in result


def test_rejects_query_missing_dataset_id_parameter():
    """A query that never references $dataset_id is rejected -- the dataset-scoping nudge."""
    cypher = "MATCH (e) RETURN e"
    with pytest.raises(CypherValidationError, match="dataset_id"):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_rejects_query_not_starting_with_a_read_clause():
    """A query not starting with MATCH/OPTIONAL MATCH/WITH/UNWIND/RETURN is rejected."""
    cypher = "// header\nMATCH (e {dataset_id: $dataset_id}) RETURN e"
    with pytest.raises(CypherValidationError):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)


def test_case_insensitive_keyword_rejection():
    """A lowercase mutation keyword is caught the same as uppercase (no case-based bypass)."""
    cypher = "match (e {dataset_id: $dataset_id}) create (x) return e"
    with pytest.raises(CypherValidationError):
        validate_generated_cypher(cypher, max_depth=3, max_result_limit=50)
