"""Path-suffix matching shared by gold-data evaluation and retrieval-time freshness logic.

Lives outside `rag.eval` (which `rag.ingestion`/`rag.retrieval`/`rag.generation`
must never import -- see `tests/unit/test_gold_data_isolation.py`) so that
`rag.retrieval.freshness`'s document-version-family resolution can reuse the
exact same suffix-matching rule `rag.eval.gold_schema` uses for
`relevant_documents`, without creating a reverse eval -> retrieval-adjacent
import or duplicating the regex logic in two places.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def path_parts(path: str) -> tuple[str, ...]:
    """Split a possibly-backslash path into POSIX-style path segments."""
    return PurePosixPath(path.replace("\\", "/")).parts


def source_matches_relevant(retrieved_source: str, relevant_path: str) -> bool:
    """Check whether `relevant_path`'s segments trail `retrieved_source`'s segments.

    E.g. "knowledge_base/security/x.md" matches a stored source of
    "data/knowledge_base/security/x.md" regardless of the "data/" root,
    without either side needing to hardcode that root.

    Parameters
    ----------
    retrieved_source : str
        A chunk's stored `source` path.
    relevant_path : str
        A shorter, relative path expected to be a trailing subsequence.

    Returns
    -------
    bool
        True if `relevant_path` is a trailing subsequence of `retrieved_source`.
    """
    retrieved_parts = path_parts(retrieved_source)
    relevant_parts = path_parts(relevant_path)
    if not relevant_parts or len(relevant_parts) > len(retrieved_parts):
        return False
    return retrieved_parts[-len(relevant_parts) :] == relevant_parts
