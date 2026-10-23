"""Hand-curated alias folding for entity names.

Case and whitespace variants are already handled by `schema.canonical_entity_id`.
This module maps known synonyms, such as "Postgres" and "PostgreSQL", to one
display name. It uses an explicit table rather than fuzzy matching, because
fuzzy matching can merge distinct entities with similar names.
"""

from __future__ import annotations

# Lowercased, whitespace-collapsed alias -> canonical display name. Edit by
# hand only. Never populate from LLM output.
_KNOWN_ALIASES: dict[str, str] = {
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "postgres database": "PostgreSQL",
    "the postgres database": "PostgreSQL",
    "postgres db": "PostgreSQL",
    "k8s": "Kubernetes",
    "opensearch": "OpenSearch",
    "elasticsearch": "OpenSearch",
}


def canonicalize_display_name(name: str) -> str:
    """Resolve `name` to its canonical display name, if a known alias exists.

    Parameters
    ----------
    name : str
        Raw entity name, as extracted.

    Returns
    -------
    str
        The alias table's canonical display name for a known alias
        (case/whitespace-insensitive lookup), or `name` unchanged if no
        alias is registered.
    """
    key = " ".join(name.split()).lower()
    return _KNOWN_ALIASES.get(key, name)
