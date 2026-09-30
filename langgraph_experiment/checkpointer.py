"""Durable checkpointing for the experimental graph: Postgres preferred, SQLite as a fallback.

`postgres_checkpointer` is the preferred backend for "the final
implementation" (per the prompt spec's production-workflow addendum):
`langgraph-checkpoint-postgres` is a real, supported, reasonably
lightweight LangGraph checkpointer (confirmed by installing it and
round-tripping an interrupt/resume/restart cycle against this project's
own local Postgres before writing this module, not assumed), and this
project already runs Postgres for everything else. `sqlite_checkpointer`
stays available as a lighter-weight fallback needing no Postgres --
`run_demo.py --backend sqlite` -- and remains what every restart/resume
test in this package uses internally (via a `tmp_path` file, never
touching the shared dev database).

Both are real, file/database-backed checkpointers, not the in-memory
default: stopping a CLI process and starting a new one against the same
target resumes exactly where it left off. Every CLI entrypoint opens and
closes its own connection per invocation (see `cli/run_demo.py`/
`cli/approval_cli.py`) rather than holding one open across calls -- each
CLI invocation *is* a fresh "process," which is the point being
demonstrated.

Checkpoint tables live in the same Postgres database/`public` schema this
project's own `documents`/`chunks`/`feedback` tables already use
(`checkpoints`/`checkpoint_blobs`/`checkpoint_writes`/
`checkpoint_migrations`, created by `PostgresSaver.setup()`) --
distinctly named, no collision, but a real production deployment would
more likely use a dedicated schema or database for full isolation; kept
simple here since a second schema/database is exactly the kind of "extra
infrastructure for appearance" the prompt spec says not to add without a
demonstrated need.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "checkpoints.sqlite"


@contextmanager
def sqlite_checkpointer(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[SqliteSaver]:
    """Open (creating if needed) the SQLite checkpoint database at `db_path`.

    Parameters
    ----------
    db_path : Path | str, optional
        Defaults to `langgraph_experiment/data/checkpoints.sqlite`
        (gitignored; see that directory's `.gitkeep`).

    Yields
    ------
    SqliteSaver
        Pass directly to `langgraph_experiment.graph.build_graph`.
    """
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(resolved)) as saver:
        yield saver


@contextmanager
def postgres_checkpointer(conn_string: str) -> Iterator[PostgresSaver]:
    """Open a Postgres-backed checkpoint connection, creating its tables if needed.

    Parameters
    ----------
    conn_string : str
        A `postgresql://` DSN -- typically `config.database_url()`, the
        same connection string `PgVectorStore` already uses.

    Yields
    ------
    PostgresSaver
        Pass directly to `langgraph_experiment.graph.build_graph`.
        `.setup()` is called every time (idempotent -- `CREATE TABLE IF
        NOT EXISTS`, confirmed directly against the installed
        `langgraph-checkpoint-postgres==3.1.2`), so a caller never needs
        to remember to run it separately first.
    """
    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()
        yield saver
