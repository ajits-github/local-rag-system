"""Durable, SQLite-backed checkpointing for the multi-agent graph.

Identical mechanism to `langgraph_experiment.checkpointer.
sqlite_checkpointer` (a real file-backed checkpointer, not the in-memory
default, so the same restart/resume story applies to the business-write
branch here too), pointed at this package's own `data/checkpoints.sqlite`
so the two experiments' demo threads never collide in one database file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "checkpoints.sqlite"


@contextmanager
def sqlite_checkpointer(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[SqliteSaver]:
    """Open (creating if needed) the SQLite checkpoint database at `db_path`.

    Parameters
    ----------
    db_path : Path | str, optional
        Defaults to `multi_agent_experiment/data/checkpoints.sqlite`
        (gitignored; see that directory's `.gitkeep`).

    Yields
    ------
    SqliteSaver
        Pass directly to `multi_agent_experiment.graph.build_graph`.
    """
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(resolved)) as saver:
        yield saver
