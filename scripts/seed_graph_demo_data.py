"""Seed the `techfusion_graph_demo` dataset: ingest sample documents, then add curated graph facts.

The relationships are hand-written, not model-extracted. Each comment next
to a relationship quotes the corpus sentence it is based on, and each one is
recorded with `extraction_method="deterministic"`. The script ingests the
documents into Postgres and writes the graph facts to Neo4j. Neo4j must be
running.

Usage:
    python scripts/seed_graph_demo_data.py [--config config/default.yaml] [--clear]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.graph.client import Neo4jClient  # noqa: E402
from rag.graph.schema import GraphProvenance, GraphRelationship  # noqa: E402
from rag.graph.upsert import GraphWriter  # noqa: E402
from rag.ingestion.pipeline import IngestionPipeline  # noqa: E402

DATASET_ID = "techfusion_graph_demo"

REAL_FILES = [
    ("architecture/system-overview.md", "architecture", "system-overview.md"),
    ("runbooks/document-service-outage.md", "runbooks", "document-service-outage.md"),
    ("policies/data-retention-policy.md", "policies", "data-retention-policy.md"),
]

# Tracked synthetic documents in data/graph_demo_docs/. They cover a SUPERSEDES
# pair and a tenant-scoped document, which the gitignored corpus does not.
DEMO_DOC_FILES = [
    ("alpha-internal-console.md", "internal_tools"),
    ("deployment-policy-v1.md", "policies"),
    ("deployment-policy-v2.md", "policies"),
]

# Each comment quotes or paraphrases the data/knowledge_base sentence it is based on.
# `document_source` names the file it comes from.
SEEDED_RELATIONSHIPS: list[dict] = [
    # system-overview.md: "Document Service owns document state."
    # system-overview.md: "PostgreSQL stores metadata and workflow state."
    dict(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="PostgreSQL",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "OCR Worker, Classifier, and Extraction Worker perform
    # asynchronous processing." / "long processing uses Kafka."
    dict(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="DEPENDS_ON",
        object_type="Technology",
        object_name="Kafka",
        document_source="architecture/system-overview.md",
    ),
    dict(
        subject_type="Service",
        subject_name="Workflow Engine",
        relationship_type="DEPENDS_ON",
        object_type="Technology",
        object_name="Kafka",
        document_source="architecture/system-overview.md",
    ),
    dict(
        subject_type="Service",
        subject_name="Notification Service",
        relationship_type="DEPENDS_ON",
        object_type="Technology",
        object_name="Kafka",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "RAG Service answers authorized internal support questions
    # from approved knowledge sources." (implies it reads persisted document/workflow state)
    dict(
        subject_type="Service",
        subject_name="RAG Service",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="PostgreSQL",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "OpenSearch holds tenant-scoped search indexes."
    dict(
        subject_type="Service",
        subject_name="RAG Service",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="OpenSearch",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "Workflow owns Document Service, Workflow Engine,
    # and Notification Service."
    dict(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="OWNED_BY",
        object_type="Team",
        object_name="Workflow",
        document_source="architecture/system-overview.md",
    ),
    dict(
        subject_type="Service",
        subject_name="Workflow Engine",
        relationship_type="OWNED_BY",
        object_type="Team",
        object_name="Workflow",
        document_source="architecture/system-overview.md",
    ),
    dict(
        subject_type="Service",
        subject_name="Notification Service",
        relationship_type="OWNED_BY",
        object_type="Team",
        object_name="Workflow",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "The Platform Engineering team owns Gateway and shared libraries."
    dict(
        subject_type="Service",
        subject_name="Gateway",
        relationship_type="OWNED_BY",
        object_type="Team",
        object_name="Platform Engineering",
        document_source="architecture/system-overview.md",
    ),
    # system-overview.md: "Trust and Compliance owns policy requirements and
    # Compliance Ledger product rules."
    dict(
        subject_type="Service",
        subject_name="Compliance Ledger",
        relationship_type="OWNED_BY",
        object_type="Team",
        object_name="Trust and Compliance",
        document_source="architecture/system-overview.md",
    ),
    # document-service-outage.md: "Document Service is owned by Workflow; SRE
    # is the platform escalation."
    # The runbook is scoped to Document Service.
    dict(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="HAS_RUNBOOK",
        object_type="Runbook",
        object_name="Document Service Outage Runbook",
        document_source="runbooks/document-service-outage.md",
    ),
    # data-retention-policy.md: retention schedule's first row explicitly covers
    # "Customer documents, OCR and extraction artifacts" is Document Service's domain.
    dict(
        subject_type="Service",
        subject_name="Document Service",
        relationship_type="GOVERNED_BY",
        object_type="Policy",
        object_name="Data Retention Policy",
        document_source="policies/data-retention-policy.md",
    ),
    # Inferred, not quoted. It creates a two-hop DEPENDS_ON chain for multi-hop
    # questions. Every other relationship paraphrases explicit corpus text.
    dict(
        subject_type="Service",
        subject_name="RAG Service",
        relationship_type="DEPENDS_ON",
        object_type="Service",
        object_name="Document Service",
        document_source="architecture/system-overview.md",
    ),
    # alpha-internal-console.md: "It depends on the Alpha Console Database..." --
    # tenant-scoped, for the cross-tenant-denial gold case.
    dict(
        subject_type="Application",
        subject_name="Alpha Internal Console",
        relationship_type="DEPENDS_ON",
        object_type="Database",
        object_name="Alpha Console Database",
        document_source="alpha-internal-console.md",
        tenant_id="tenant_alpha",
        allowed_roles=["tenant_alpha_operator"],
    ),
]


def main() -> None:
    """CLI entrypoint: ingest the real files, then seed hand-curated domain relationships."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--clear", action="store_true", help="Clear the demo dataset first")
    parser.add_argument(
        "--knowledge-base-root",
        default="data/knowledge_base",
        help="Root directory the REAL_FILES paths are relative to",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = config.model_copy(deep=True)
    config.graph.enabled = True
    config.graph.extraction.enabled = True
    config.graph.extraction.llm_extraction_enabled = (
        False  # deterministic-only; see module docstring
    )

    graph_client = Neo4jClient(config)
    if not graph_client.health_check():
        print("ERROR: Neo4j is not reachable.", file=sys.stderr)
        sys.exit(1)

    pipeline = IngestionPipeline(config, graph_client=graph_client)

    if args.clear:
        # Clear both stores. An unchanged checksum makes ingest_file skip graph
        # extraction, so clearing only the graph would leave it empty.
        pipeline.clear_dataset(DATASET_ID)
        print(f"Cleared existing Postgres chunks and graph facts for dataset '{DATASET_ID}'")
    root = Path(args.knowledge_base_root)
    document_ids: dict[str, str] = {}
    for relative_path, category, _filename in REAL_FILES:
        full_path = root / relative_path
        if not full_path.is_file():
            print(f"SKIP (not found): {full_path}", file=sys.stderr)
            continue
        result = pipeline.ingest_file(full_path, DATASET_ID, category=category)
        document_ids[relative_path] = result["document_id"]
        print(f"Ingested {relative_path} -> {result}")

    demo_docs_root = Path(__file__).resolve().parents[1] / "data" / "graph_demo_docs"
    for filename, category in DEMO_DOC_FILES:
        full_path = demo_docs_root / filename
        result = pipeline.ingest_file(full_path, DATASET_ID, category=category)
        document_ids[filename] = result["document_id"]
        print(f"Ingested {filename} -> {result}")

    writer = GraphWriter(graph_client)
    written = 0
    for rel in SEEDED_RELATIONSHIPS:
        rel = dict(rel)
        document_source = rel.pop("document_source")
        tenant_id = rel.pop("tenant_id", None)
        allowed_roles = rel.pop("allowed_roles", None)
        document_id = document_ids.get(document_source)
        provenance = GraphProvenance(
            source_document_id=document_id or "seed-script",
            dataset_id=DATASET_ID,
            tenant_id=tenant_id,
            allowed_roles=allowed_roles,
            extraction_method="deterministic",
            confidence=1.0,
            document_checksum="seed-script",
        )
        writer.upsert_relationship(GraphRelationship(provenance=provenance, **rel))
        written += 1
    print(f"Seeded {written} hand-curated relationships.")

    graph_client.close()


if __name__ == "__main__":
    main()
