"""Backend-agnostic RDF store abstraction (docs/ontology.md section 8).

`RDFStoreBackend` is the small surface this milestone actually needs
from an RDF store: load Turtle files, run a SPARQL query, report how
many triples are loaded, and serialize back out. `RdflibStore` is the
only implementation (an in-process rdflib.Graph, no server process),
per this milestone's own instruction to use RDFLib first and keep
things lightweight. Every demo, CLI script, and test in this package
is written against `RDFStoreBackend`'s surface, not against
rdflib-specific APIs directly (with the single, documented exception of
`.graph`, exposed for the two libraries, owlrl and pyshacl, that
need a real `rdflib.Graph` object, since neither offers a SPARQL-HTTP
mode this milestone would otherwise need).

A future `FusekiStore`/`GraphDBStore` implementing the same `load` /
`query` / `__len__` / `serialize` surface over a real SPARQL HTTP
endpoint (python's `SPARQLWrapper`, or rdflib's own `SPARQLStore`
graph implementation) would let every CLI/test here run unmodified
against a real triple-store server, the "keep backend abstraction
clean enough that the ontology is not tied unnecessarily to one
implementation" requirement. Not built in this milestone: no concrete
learning need for a real server was identified (see
`docs/ontology.md`'s "RDF store" section for the reasoning, which
mirrors this codebase's own repeated "no infrastructure without a
demonstrated requirement now" pattern, e.g. Redis and MCP in the main
project).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from rdflib import Graph
from rdflib.query import Result
from rdflib.term import Identifier


class RDFStoreBackend(Protocol):
    """The minimal surface a backend must provide for this milestone."""

    def load(self, path: str | Path) -> None:
        """Parse one Turtle file into this store."""
        ...

    def query(self, sparql: str, init_bindings: Mapping[str, Identifier] | None = None) -> Result:
        """Run one SPARQL query and return rdflib's `Result` object."""
        ...

    def __len__(self) -> int:
        """Return the number of triples currently held."""
        ...

    def serialize(self, destination: str | Path, fmt: str = "turtle") -> None:
        """Write every held triple to `destination` in `fmt`."""
        ...


class RdflibStore:
    """In-memory, single-process RDF store backed by `rdflib.Graph`.

    No persistence beyond what a caller explicitly serializes, and no
    server process: this is the lightweight default this milestone's
    own instructions call for. Namespace prefix bindings loaded from
    each parsed Turtle file's own `@prefix` declarations are kept, so
    query results and serialized output stay in the same, readable
    `tech:`/`inst:` form the source files use.
    """

    def __init__(self) -> None:
        self.graph: Graph = Graph()

    def load(self, path: str | Path) -> None:
        """Parse one Turtle file into this store.

        Parameters
        ----------
        path : str | Path
            Path to a `.ttl` file.
        """
        self.graph.parse(str(path), format="turtle")

    def load_many(self, paths: Iterable[str | Path]) -> None:
        """Parse several Turtle files into this store, in order.

        Parameters
        ----------
        paths : Iterable[str | Path]
            Paths to `.ttl` files, parsed one after another into the
            same graph.
        """
        for path in paths:
            self.load(path)

    def query(self, sparql: str, init_bindings: Mapping[str, Identifier] | None = None) -> Result:
        """Run one SPARQL query against this store.

        Parameters
        ----------
        sparql : str
            A SPARQL 1.1 query string.
        init_bindings : Mapping[str, Identifier] | None, optional
            Pre-bound variable values (rdflib's `initBindings`), the
            safe way to parameterize a query with a caller-supplied
            value without string-interpolating it into the query text.

        Returns
        -------
        rdflib.query.Result
            Iterable of result rows (for SELECT) or a boolean (for ASK).
        """
        return self.graph.query(sparql, initBindings=init_bindings or {})

    def __len__(self) -> int:
        """Return the number of triples currently held."""
        return len(self.graph)

    def serialize(self, destination: str | Path, fmt: str = "turtle") -> None:
        """Write every held triple to `destination` in `fmt`.

        Parameters
        ----------
        destination : str | Path
            Output file path.
        fmt : str, optional
            An rdflib serialization format name, by default "turtle".
        """
        self.graph.serialize(destination=str(destination), format=fmt)

    def copy(self) -> RdflibStore:
        """Return an independent, deep copy of this store.

        Needed because RDFS/OWL-RL materialization (`reasoning.py`)
        inserts inferred triples IN PLACE into whatever graph it is
        given, exactly like a real forward-chaining reasoner would.
        Every demo/test that wants to compare "before" and "after"
        inference takes a copy first, so a destructive step never
        mutates a store another demo/test still needs clean.

        Returns
        -------
        RdflibStore
            A new store with the same triples and namespace bindings,
            sharing no mutable state with this one.
        """
        new_store = RdflibStore()
        for triple in self.graph:
            new_store.graph.add(triple)
        for prefix, namespace in self.graph.namespaces():
            new_store.graph.bind(prefix, namespace)
        return new_store


def load_store(paths: Iterable[str | Path]) -> RdflibStore:
    """Build a fresh `RdflibStore` and load the given Turtle files into it.

    Parameters
    ----------
    paths : Iterable[str | Path]
        Paths to `.ttl` files, loaded in order.

    Returns
    -------
    RdflibStore
        A new store holding every triple from every given file.
    """
    store = RdflibStore()
    store.load_many(paths)
    return store
