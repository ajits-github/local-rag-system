"""Text checks on LLM-generated Cypher. Best-effort, not the security boundary.

The checks reject multiple statements, comments, write and admin keywords,
unbounded or over-deep traversals, and queries that omit `$dataset_id`. A
keyword denylist cannot be proven complete, so two other layers enforce
the guarantee:

1. Execution goes through `Neo4jClient.run_read`, which runs in a read
   transaction. The server rejects writes in that mode.
2. `GraphRetriever` re-checks every returned fact against the caller's
   `AuthorizationContext`, whatever the generated `WHERE` clause did.

The generated text is never trusted for tenant or role isolation.
"""

from __future__ import annotations

import re

_MULTI_STATEMENT_RE = re.compile(r";")
_COMMENT_RE = re.compile(r"//|/\*")
_STARTS_WITH_READ_RE = re.compile(r"^\s*(MATCH|OPTIONAL MATCH|WITH|UNWIND|RETURN)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
_DATASET_PARAM_RE = re.compile(r"\$dataset_id\b")

# Any whole-word match rejects the query. CALL is banned outright, not only
# apoc.* or dbms.* procedures, because every procedure call starts with CALL.
_FORBIDDEN_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "CALL",
    "LOAD",
    "FOREACH",
    "START",
    "USE",
    "SHOW",
    "TERMINATE",
    "GRANT",
    "REVOKE",
    "DENY",
    "INDEX",
    "CONSTRAINT",
)
_FORBIDDEN_KEYWORD_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)

# Matches a variable-length relationship pattern's star quantifier, e.g.
# "*", "*3", "*1..3", "*..3", "*2..". Group 2 (the upper bound) is what
# matters: absent means unbounded.
_VAR_LENGTH_RE = re.compile(r"\*\s*(\d*)\s*(?:\.\.\s*(\d*))?")


class CypherValidationError(ValueError):
    """Raised when a generated Cypher query fails validation.

    The message never contains the full query text. Callers audit-log
    `str(exc)`, and the query may have been influenced by injected content.
    """


def validate_generated_cypher(cypher: str, *, max_depth: int, max_result_limit: int) -> str:
    """Validate and normalize one LLM-generated Cypher query.

    Parameters
    ----------
    cypher : str
        The raw generated query text.
    max_depth : int
        Upper bound every variable-length relationship pattern's own
        upper bound must not exceed (`config.graph.cypher_generation.max_depth`).
    max_result_limit : int
        The `LIMIT` value stamped onto the returned query, replacing
        whatever (if anything) the generated text itself specified
        (`config.graph.cypher_generation.max_result_limit`).

    Returns
    -------
    str
        The validated query, with any existing `LIMIT` clause stripped
        and a single trusted `LIMIT <max_result_limit>` appended.

    Raises
    ------
    CypherValidationError
        If the query is empty, contains more than one statement, contains
        a comment, does not begin with a read clause, contains a
        forbidden (write/admin/procedure) keyword, contains an unbounded
        or over-deep variable-length relationship pattern, or contains no
        `$dataset_id` parameter reference.
    """
    stripped = cypher.strip().rstrip(";").strip()
    if not stripped:
        raise CypherValidationError("Generated Cypher is empty")

    if _MULTI_STATEMENT_RE.search(stripped):
        raise CypherValidationError("Generated Cypher contains more than one statement")

    if _COMMENT_RE.search(stripped):
        raise CypherValidationError("Generated Cypher contains a comment")

    if not _STARTS_WITH_READ_RE.match(stripped):
        raise CypherValidationError(
            "Generated Cypher must start with MATCH/OPTIONAL MATCH/WITH/UNWIND/RETURN"
        )

    forbidden = _FORBIDDEN_KEYWORD_RE.search(stripped)
    if forbidden:
        raise CypherValidationError(
            f"Generated Cypher contains a forbidden clause: {forbidden.group(1)}"
        )

    for match in _VAR_LENGTH_RE.finditer(stripped):
        upper_bound_str = match.group(2) if match.group(2) is not None else match.group(1)
        if not upper_bound_str:
            raise CypherValidationError(
                "Generated Cypher contains an unbounded variable-length traversal"
            )
        if int(upper_bound_str) > max_depth:
            raise CypherValidationError(
                f"Generated Cypher's traversal depth {upper_bound_str} "
                f"exceeds max_depth={max_depth}"
            )

    if not _DATASET_PARAM_RE.search(stripped):
        raise CypherValidationError("Generated Cypher must reference the $dataset_id parameter")

    normalized = _LIMIT_RE.sub("", stripped).strip()
    return f"{normalized}\nLIMIT {max_result_limit}"
