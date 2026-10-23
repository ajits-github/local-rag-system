"""Measure graph query latency on a synthetic graph of `Service` nodes.

Generates `--nodes` Service nodes with random DEPENDS_ON edges, then times
direct lookup, 1-hop, 2-hop, max-depth traversal, and shortest path. Data is
written to `benchmark_<nodes>` and deleted afterward unless `--keep` is set.

Usage:
    python scripts/benchmark_graph.py --nodes 1000 [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.graph.client import Neo4jClient  # noqa: E402
from rag.graph.retrieval import GraphRetriever  # noqa: E402
from rag.graph.schema import GraphProvenance, GraphRelationship  # noqa: E402
from rag.graph.upsert import GraphWriter  # noqa: E402


def _generate_graph(writer: GraphWriter, dataset_id: str, node_count: int, avg_degree: int) -> None:
    """Write `node_count` Service nodes with `avg_degree` random outgoing DEPENDS_ON edges each."""
    names = [f"service-{i}" for i in range(node_count)]
    provenance_template = dict(
        source_document_id="benchmark-doc",
        dataset_id=dataset_id,
        extraction_method="deterministic",
        confidence=1.0,
        document_checksum="benchmark",
    )
    written = 0
    for i, name in enumerate(names):
        targets = random.sample(names, k=min(avg_degree, node_count - 1))
        for target in targets:
            if target == name:
                continue
            writer.upsert_relationship(
                GraphRelationship(
                    subject_type="Service",
                    subject_name=name,
                    relationship_type="DEPENDS_ON",
                    object_type="Service",
                    object_name=target,
                    provenance=GraphProvenance(**provenance_template),
                )
            )
            written += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{node_count} nodes written ({written} relationships so far)")


def _time_calls(fn, n_calls: int) -> dict[str, float]:
    """Run `fn()` `n_calls` times and return latency percentile stats in milliseconds."""
    latencies = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {
        "min_ms": latencies[0],
        "p50_ms": statistics.median(latencies),
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1],
        "max_ms": latencies[-1],
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--avg-degree", type=int, default=4)
    parser.add_argument("--samples", type=int, default=20, help="Query samples per measurement")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--keep", action="store_true", help="Do not delete the benchmark data afterward"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_id = f"benchmark_{args.nodes}"
    client = Neo4jClient(config)
    if not client.health_check():
        print("ERROR: Neo4j is not reachable.", file=sys.stderr)
        sys.exit(1)

    writer = GraphWriter(client)
    retriever = GraphRetriever(config, client)

    print(
        f"Generating {args.nodes} nodes (avg out-degree {args.avg_degree}) "
        f"in dataset '{dataset_id}'..."
    )
    t0 = time.perf_counter()
    _generate_graph(writer, dataset_id, args.nodes, args.avg_degree)
    generation_s = time.perf_counter() - t0
    print(f"Generation took {generation_s:.1f}s")

    names = [f"service-{i}" for i in range(args.nodes)]

    print("\nDirect lookup (resolve_entities, exact name match):")
    stats = _time_calls(
        lambda: retriever.resolve_entities(
            random.choice(names), dataset_id=dataset_id, auth=None, limit=1
        ),
        args.samples,
    )
    print(f"  {stats}")

    print("\n1-hop traversal (get_neighbors, max_hops=1):")
    stats = _time_calls(
        lambda: retriever.get_neighbors(
            random.choice(names), dataset_id=dataset_id, auth=None, max_hops=1
        ),
        args.samples,
    )
    print(f"  {stats}")

    print("\n2-hop traversal (get_neighbors, max_hops=2):")
    stats = _time_calls(
        lambda: retriever.get_neighbors(
            random.choice(names), dataset_id=dataset_id, auth=None, max_hops=2
        ),
        args.samples,
    )
    print(f"  {stats}")

    print(
        f"\n3-hop traversal (get_neighbors, max_hops={config.graph.traversal.max_traversal_depth}):"
    )
    stats = _time_calls(
        lambda: retriever.get_neighbors(
            random.choice(names),
            dataset_id=dataset_id,
            auth=None,
            max_hops=config.graph.traversal.max_traversal_depth,
        ),
        args.samples,
    )
    print(f"  {stats}")

    print("\nBounded shortest-path query (find_path, two random nodes):")
    stats = _time_calls(
        lambda: retriever.find_path(
            random.choice(names), random.choice(names), dataset_id=dataset_id, auth=None
        ),
        args.samples,
    )
    print(f"  {stats}")

    if not args.keep:
        print(f"\nCleaning up dataset '{dataset_id}'...")
        writer.delete_facts_by_dataset(dataset_id)
    client.close()


if __name__ == "__main__":
    main()
