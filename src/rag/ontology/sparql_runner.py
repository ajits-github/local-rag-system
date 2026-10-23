"""Load and run the representative `.rq` files under `queries/sparql/` (docs/ontology.md section 7).

Kept deliberately tiny: this milestone's queries are plain text files,
not a templating system, so running one is just reading the file and
calling `store.query`. The value this module adds is turning rdflib's
`Result` rows into plain, JSON-printable dicts for the CLI demos and
tests, since a raw `rdflib.query.Result` row holds rdflib term objects
(`URIRef`/`Literal`) that do not print or compare the way a reader
expects.
"""

from __future__ import annotations

from pathlib import Path

from rag.ontology.store import RdflibStore


def load_query_text(path: str | Path) -> str:
    """Read one `.rq` file's text, comments and all.

    Parameters
    ----------
    path : str | Path
        Path to a `.rq` file.

    Returns
    -------
    str
        The file's full text.
    """
    return Path(path).read_text(encoding="utf-8")


def run_query_file(store: RdflibStore, path: str | Path) -> list[dict[str, str | None]]:
    """Run one `.rq` file's query against `store` and return plain-dict rows.

    Parameters
    ----------
    store : RdflibStore
        The graph to query.
    path : str | Path
        Path to a `.rq` file.

    Returns
    -------
    list[dict[str, str | None]]
        One dict per result row, keyed by SPARQL variable name
        (without the leading `?`), values stringified (`None` for an
        unbound `OPTIONAL` variable).
    """
    from typing import cast

    from rdflib.query import ResultRow

    sparql = load_query_text(path)
    result = store.query(sparql)
    result_rows = cast("list[ResultRow]", list(result))
    rows: list[dict[str, str | None]] = []
    for row in result_rows:
        row_dict: dict[str, str | None] = {}
        for var_name in result.vars or []:
            value = row[var_name]
            row_dict[str(var_name)] = str(value) if value is not None else None
        rows.append(row_dict)
    return rows


def format_rows(rows: list[dict[str, str | None]]) -> str:
    """Render query result rows as an aligned, readable text table.

    Parameters
    ----------
    rows : list[dict[str, str | None]]
        Rows as returned by `run_query_file`.

    Returns
    -------
    str
        A multi-line string: one header line, one line per row, `"(no
        results)"` if `rows` is empty.
    """
    if not rows:
        return "(no results)"
    columns = list(rows[0].keys())
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(_short(row.get(col)) for col in columns))
    return "\n".join(lines)


def _short(value: str | None) -> str:
    """Shorten a full IRI to its final path/fragment segment for display only."""
    if value is None:
        return "-"
    for separator in ("#", "/"):
        if separator in value:
            value = value.rsplit(separator, 1)[-1]
    return value
