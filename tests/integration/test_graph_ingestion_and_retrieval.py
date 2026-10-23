"""End-to-end graph-ingestion and retrieval tests against a real Neo4j instance.

Self-skips without a reachable Neo4j (`require_neo4j`) or Postgres
(`require_postgres`), mirroring `test_authorization_isolation.py`'s
self-contained convention: synthetic documents ingested into a fresh
`dataset_id` per test, never depending on the real (gitignored)
`data/knowledge_base` corpus being present.

Deterministic-only extraction throughout (`llm_extraction_enabled=False`):
these tests prove the graph-ingestion *mechanism* (deterministic facts,
consistency on re-ingestion/deletion, bounded traversal, authorization),
not an LLM's extraction judgment -- domain relationships needed for the
traversal/authorization tests are written directly via `GraphWriter`,
the same "test the mechanism, not the LLM" reasoning
`test_specialized_tool_gold_fixtures.py` already establishes for the
agent's local-tool tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.graph.client import Neo4jClient
from rag.graph.pipeline import GraphRAGPipeline
from rag.graph.retrieval import GraphRetriever
from rag.graph.schema import GraphProvenance, GraphRelationship
from rag.graph.upsert import GraphWriter
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.pipeline import RetrievalPipeline


def _graph_config(config):
    """Return a copy of `config` with graph enabled (deterministic extraction only)."""
    graphed = config.model_copy(deep=True)
    graphed.graph.enabled = True
    graphed.graph.extraction.enabled = True
    graphed.graph.extraction.llm_extraction_enabled = False
    return graphed


def _write_doc(tmp_path: Path, name: str, frontmatter: dict, body: str) -> Path:
    lines = ["---"]
    for key, value in frontmatter.items():
        lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append("")
    lines.append(body)
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _seed_relationship(
    client: Neo4jClient, dataset_id: str, document_id: str, subject: str, rel_type: str, obj: str
) -> None:
    """Write one hand-constructed relationship directly, bypassing LLM extraction."""
    writer = GraphWriter(client)
    provenance = GraphProvenance(
        source_document_id=document_id,
        source_chunk_id=f"{document_id}_0",
        dataset_id=dataset_id,
        extraction_method="deterministic",
        confidence=1.0,
        document_checksum="test-checksum",
    )
    writer.upsert_relationship(
        GraphRelationship(
            subject_type="Service",
            subject_name=subject,
            relationship_type=rel_type,
            object_type="Database" if obj == "PostgreSQL" else "Service",
            object_name=obj,
            provenance=provenance,
        )
    )


def test_deterministic_extraction_creates_document_and_category_nodes(
    require_postgres, require_neo4j, config, tmp_path: Path
):
    """Ingesting a categorized document creates a Document node and a PART_OF -> Category edge."""
    ns = f"pytest-graph-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        pipeline = IngestionPipeline(graphed, graph_client=client)
        path = _write_doc(tmp_path, "doc.md", {}, "Some ordinary document content.")
        result = pipeline.ingest_file(path, ns, category="architecture")
        assert result["changed"]

        rows = client.run_read(
            "MATCH (d:Document)-[:PART_OF]->(c:Category) WHERE d.dataset_id = $ds "
            "RETURN d.document_id AS document_id, c.name AS category",
            {"ds": ns},
        )
        assert len(rows) == 1
        assert rows[0]["document_id"] == result["document_id"]
        assert rows[0]["category"] == "architecture"
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_reingesting_a_changed_document_replaces_its_relationships_without_duplicates(
    require_postgres, require_neo4j, config, tmp_path: Path
):
    """A changed document's stale relationship is gone after re-ingestion; nothing accumulates."""
    ns = f"pytest-graph-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        pipeline = IngestionPipeline(graphed, graph_client=client)
        path = _write_doc(tmp_path, "doc.md", {}, "Version one.")
        result = pipeline.ingest_file(path, ns)
        document_id = result["document_id"]
        _seed_relationship(client, ns, document_id, "Old Service", "DEPENDS_ON", "Old Database")

        rows_before = client.run_read(
            "MATCH ()-[r:DEPENDS_ON]->() WHERE r.dataset_id = $ds RETURN count(r) AS n", {"ds": ns}
        )
        assert rows_before[0]["n"] == 1

        # Re-ingest with changed content -> deterministic re-extraction runs again,
        # which must delete facts sourced from this document_id before writing fresh ones.
        path.write_text("Version two, different content entirely.", encoding="utf-8")
        result2 = pipeline.ingest_file(path, ns)
        assert result2["changed"]
        assert result2["document_id"] == document_id

        rows_after = client.run_read(
            "MATCH ()-[r:DEPENDS_ON]->() WHERE r.dataset_id = $ds RETURN count(r) AS n", {"ds": ns}
        )
        # The hand-seeded relationship (sourced from this same document_id) is gone;
        # deterministic re-extraction wrote no new DEPENDS_ON facts of its own.
        assert rows_after[0]["n"] == 0
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_deleting_a_document_removes_its_graph_facts(
    require_postgres, require_neo4j, config, tmp_path: Path
):
    """clear_dataset removes both the Postgres chunks and every graph fact for that dataset."""
    ns = f"pytest-graph-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        pipeline = IngestionPipeline(graphed, graph_client=client)
        path = _write_doc(tmp_path, "doc.md", {}, "Content to be deleted.")
        pipeline.ingest_file(path, ns)

        rows_before = client.run_read(
            "MATCH (n) WHERE n.dataset_id = $ds RETURN count(n) AS n", {"ds": ns}
        )
        assert rows_before[0]["n"] > 0

        pipeline.clear_dataset(ns)

        rows_after = client.run_read(
            "MATCH (n) WHERE n.dataset_id = $ds RETURN count(n) AS n", {"ds": ns}
        )
        assert rows_after[0]["n"] == 0
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_dependents_of_traversal_finds_a_hand_seeded_dependency(
    require_postgres, require_neo4j, config, tmp_path: Path
):
    """search_knowledge_graph resolves a 'what depends on X' question via real bounded traversal."""
    ns = f"pytest-graph-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        pipeline = IngestionPipeline(graphed, graph_client=client)
        path = _write_doc(tmp_path, "doc.md", {}, "Some document that mentions services.")
        result = pipeline.ingest_file(path, ns)
        _seed_relationship(
            client, ns, result["document_id"], "Document Service", "DEPENDS_ON", "PostgreSQL"
        )

        retriever = GraphRetriever(graphed, client)
        search_result = retriever.search("What depends on PostgreSQL?", dataset_id=ns, auth=None)
        assert search_result.intent == "dependents_of"
        assert any("Document Service" in fact.text for fact in search_result.facts)
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
        client.close()


def test_graphrag_pipeline_graph_only_cites_a_real_document_source(
    require_postgres, require_ollama, require_neo4j, config, tmp_path: Path
):
    """GraphRAGPipeline's graph_only strategy answers and cites a real, resolvable document path."""
    ns = f"pytest-graph-{uuid.uuid4()}"
    graphed = _graph_config(config)
    client = Neo4jClient(graphed)
    try:
        pipeline = IngestionPipeline(graphed, graph_client=client)
        path = _write_doc(tmp_path, "doc.md", {}, "A document about Document Service.")
        result = pipeline.ingest_file(path, ns)
        _seed_relationship(
            client, ns, result["document_id"], "Document Service", "DEPENDS_ON", "PostgreSQL"
        )

        retrieval_pipeline = RetrievalPipeline(graphed)
        retriever = GraphRetriever(graphed, client, llm=retrieval_pipeline.llm)
        graphrag = GraphRAGPipeline(
            graphed, retrieval_pipeline, graph_client=client, graph_retriever=retriever
        )
        answer = graphrag.answer(
            "What does Document Service depend on?",
            filters={"dataset_id": ns},
            strategy_override="graph_only",
        )
        assert answer["sources"]
        assert all(s["origin"] == "graph" for s in answer["sources"])
        assert any(s["source"] == str(path) for s in answer["sources"])
        graphrag.close()
    finally:
        client.run_write("MATCH (n) WHERE n.dataset_id = $ds DETACH DELETE n", {"ds": ns})
