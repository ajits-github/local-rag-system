"""Create the Neo4j uniqueness constraints and indexes. Safe to re-run.

Usage:
    python scripts/init_graph_db.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.graph.client import Neo4jClient  # noqa: E402
from rag.graph.constraints import build_all_statements  # noqa: E402


def main() -> None:
    """CLI entrypoint: build and run every constraint/index statement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.graph.enabled:
        print(
            "graph.enabled is false in the loaded config; proceeding anyway "
            "(this script only touches Neo4j, never gated by that flag itself)."
        )

    client = Neo4jClient(config)
    try:
        if not client.health_check():
            print(
                "ERROR: Neo4j is not reachable. Is `docker compose -f docker-compose.yml "
                "-f docker-compose.graph.yml up -d neo4j` running?",
                file=sys.stderr,
            )
            sys.exit(1)
        statements = build_all_statements()
        for statement in statements:
            client.run_write(statement)
        print(
            f"Applied {len(statements)} constraint/index statements to Neo4j "
            f"database '{config.graph.neo4j.database}'."
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
