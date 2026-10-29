"""Unit tests for `GraphRetriever.find_path`'s relationship-level authorization.

Regression for GRAPH-006: the shortest-path template's Python-side
re-check only ever inspected `path.nodes`, never `path.relationships`,
even though a relationship carries its own `allowed_roles`/`trust_level`/
`source_document_id` (see `graph/upsert.py`). Two public entities
connected only by a relationship sourced from a restricted document
must never have that relationship disclosed to a caller unauthorized for
it, even when both endpoint nodes are themselves public.

Uses a fake `Neo4jClient` (no real Neo4j); `dict(node)`/`.labels` are the
only Neo4j `Node` properties `GraphRetriever` actually relies on, and a
plain `dict` already satisfies `dict(relationship)` for a relationship.
"""

from __future__ import annotations

from rag.config import load_config
from rag.graph.retrieval import GraphRetriever
from rag.retrieval.authorization import AuthorizationContext


class _FakeNode(dict):
    """Minimal stand-in for a Neo4j `Node`: a dict with a `.labels` attribute."""

    def __init__(self, labels: list[str], **properties: object) -> None:
        super().__init__(**properties)
        self.labels = labels


class _FakePath:
    """Minimal stand-in for a Neo4j `Path`: `.nodes` and `.relationships`."""

    def __init__(self, nodes: list[_FakeNode], relationships: list[dict]) -> None:
        self.nodes = nodes
        self.relationships = relationships


class _SequencedFakeClient:
    """Returns canned `run_read` responses in call order; records nothing else."""

    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = responses
        self.call_count = 0

    def run_read(self, cypher: str, params: dict) -> list[dict]:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def _entity_resolution_row(canonical_id: str, name: str) -> dict:
    return {
        "e": _FakeNode(["Service"], canonical_id=canonical_id, name=name),
        "entity_labels": ["Service"],
    }


def _build_client(*, relationship_props: dict) -> _SequencedFakeClient:
    """Two public endpoint nodes, connected by one relationship with `relationship_props`."""
    node_a = _FakeNode(["Service"], canonical_id="svc-a", name="Alpha API")
    node_b = _FakeNode(["Service"], canonical_id="svc-b", name="Alpha Database")
    path = _FakePath(nodes=[node_a, node_b], relationships=[relationship_props])
    return _SequencedFakeClient(
        [
            [_entity_resolution_row("svc-a", "Alpha API")],
            [_entity_resolution_row("svc-b", "Alpha Database")],
            [{"path": path}],
        ]
    )


def test_a_restricted_relationship_between_two_public_entities_is_not_disclosed():
    """A dependency edge sourced from an admin-only document stays hidden from a non-admin."""
    client = _build_client(
        relationship_props={
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_admin"],
            "trust_level": None,
            "source_document_id": "admin-only-doc",
        }
    )
    retriever = GraphRetriever(load_config(), client)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_viewer"])

    result = retriever.find_path("Alpha API", "Alpha Database", dataset_id="ds", auth=auth)

    assert result.facts == []


def test_an_authorized_relationship_between_two_public_entities_is_disclosed():
    """The same path is returned once the caller's role is actually authorized for the edge."""
    client = _build_client(
        relationship_props={
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_admin"],
            "trust_level": None,
            "source_document_id": "admin-only-doc",
        }
    )
    retriever = GraphRetriever(load_config(), client)
    auth = AuthorizationContext(tenant_id="tenant_alpha", roles=["tenant_alpha_admin"])

    result = retriever.find_path("Alpha API", "Alpha Database", dataset_id="ds", auth=auth)

    assert len(result.facts) == 1
    assert "Alpha API" in result.facts[0].text
    assert "Alpha Database" in result.facts[0].text


def test_an_unrestricted_relationship_is_disclosed_with_no_identity_at_all():
    """`auth=None` is fully unrestricted, matching every other graph authorization check."""
    client = _build_client(
        relationship_props={
            "tenant_id": "tenant_alpha",
            "allowed_roles": ["tenant_alpha_admin"],
            "trust_level": None,
            "source_document_id": "admin-only-doc",
        }
    )
    retriever = GraphRetriever(load_config(), client)

    result = retriever.find_path("Alpha API", "Alpha Database", dataset_id="ds", auth=None)

    assert len(result.facts) == 1
