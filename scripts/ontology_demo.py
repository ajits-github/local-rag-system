"""Runnable demos for the ontology/semantic-layer milestone (`rag.ontology`).

Each `--demo` name below corresponds to one section of `docs/ontology.md`.
No external services needed (Neo4j/Postgres/Ollama); everything runs
against the bundled `data/ontology/*.ttl` files via an in-memory
`rdflib.Graph`. Install the `ontology` extra first: `pip install -e .[ontology]`.

Usage:
    python scripts/ontology_demo.py --demo all
    python scripts/ontology_demo.py --demo rdfs
    python scripts/ontology_demo.py --demo domain-range-pitfall
    python scripts/ontology_demo.py --demo owl
    python scripts/ontology_demo.py --demo shacl
    python scripts/ontology_demo.py --demo sparql
    python scripts/ontology_demo.py --demo property-graph-vs-rdf
    python scripts/ontology_demo.py --demo interop
    python scripts/ontology_demo.py --demo auth
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ontology import paths  # noqa: E402
from rag.ontology.authorization import (  # noqa: E402
    fetch_entity_auth_properties,
    is_entity_authorized,
)
from rag.ontology.interoperability import (  # noqa: E402
    find_equivalent_external_classes,
    summarize_entity_resolution,
)
from rag.ontology.reasoning import (  # noqa: E402
    find_disjoint_class_violations,
    inferred_only_triples,
    materialize_owl_rl,
    materialize_rdfs,
)
from rag.ontology.sparql_runner import format_rows, run_query_file  # noqa: E402
from rag.ontology.store import RdflibStore, load_store  # noqa: E402
from rag.ontology.validation import run_shacl_validation  # noqa: E402
from rag.retrieval.authorization import AuthorizationContext  # noqa: E402

TECH = "http://techfusion.example/ontology/v1#"
INST = "http://techfusion.example/graph/techfusion_graph_demo/"


def _short(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            iri = iri.rsplit(sep, 1)[-1]
    return iri


def demo_rdfs() -> None:
    """Explicit vs. inferred facts under RDFS closure (docs/ontology.md section 4)."""
    before = load_store([paths.ONTOLOGY_V1, paths.INSTANCES_GRAPH_DEMO])
    print(f"Explicit triples (ontology + instance data): {len(before)}")

    after = before.copy()
    report = materialize_rdfs(after)
    print(f"Triples after RDFS closure: {report.total_triple_count}")
    print(f"Newly inferred triples: {report.inferred_triple_count}")

    print()
    print("Example: is PostgreSQL explicitly or only implicitly a Service?")
    explicit_q = f"""
    PREFIX tech: <{TECH}>
    PREFIX inst: <{INST}>
    ASK {{ inst:database-postgresql a tech:Service }}
    """
    print("  Explicitly asserted (tech:Service), before RDFS:", bool(before.query(explicit_q)))
    print(
        "  True after RDFS closure (via tech:DatabaseService rdfs:subClassOf tech:Service):",
        bool(after.query(explicit_q)),
    )

    from rdflib import URIRef
    from rdflib.namespace import RDF, RDFS

    new_triples = inferred_only_triples(before, after)
    # Every RDFS closure also mechanically entails "X a rdfs:Resource" for
    # every term X used anywhere in the graph (RDFS rules rdfs4a/rdfs4b):
    # true but content-free noise for a teaching demo, filtered out here so
    # the printed examples are the ones this ontology's own subclass
    # declarations actually produced.
    type_triples = [
        t
        for t in new_triples
        if t[1] == RDF.type
        and isinstance(t[0], URIRef)
        and t[2] != RDFS.Resource
        and str(t[0]).startswith("http://techfusion.example/")
    ]
    print()
    print(
        f"Of {len(new_triples)} newly inferred triples, {len(type_triples)} are new, "
        "non-trivial rdf:type facts"
    )
    print(
        "(excluding the content-free 'X a rdfs:Resource' noise every RDFS "
        "closure also adds for every term)."
    )
    print("First 5:")
    for subject, _predicate, obj in list(type_triples)[:5]:
        print(f"  {subject} a {obj}")


TOY_ONTOLOGY = """
@prefix ex: <http://example.org/toy#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:worksAt a rdf:Property ;
    rdfs:domain ex:Person .

ex:ACMECorp ex:worksAt ex:SomeVendor .
"""


def demo_domain_range_pitfall() -> None:
    """Reproduce the RDFS domain/range inference-trigger pitfall, cleanly and in isolation.

    See `data/ontology/schema/techfusion_ontology_v1.ttl`'s own comment on
    `tech:dependsOn` for the REAL version of this bug, found and fixed in
    this ontology (a narrow `rdfs:domain tech:Service` there caused
    exactly this effect once real data included an `Application`-subject
    `dependsOn` triple; see `KNOWN_ISSUES.md`'s ONT-001 entry).
    """
    store = RdflibStore()
    store.graph.parse(data=TOY_ONTOLOGY, format="turtle")

    q = """
    PREFIX ex: <http://example.org/toy#>
    ASK { ex:ACMECorp a ex:Person }
    """
    print("Before RDFS closure, is ex:ACMECorp (a company) typed ex:Person?", bool(store.query(q)))

    materialize_rdfs(store)
    print("After RDFS closure, is ex:ACMECorp typed ex:Person?", bool(store.query(q)))
    print()
    print("Nothing here asserted 'ex:ACMECorp a ex:Person' directly. The single axiom")
    print("'ex:worksAt rdfs:domain ex:Person' combined with the one instance triple")
    print("'ex:ACMECorp ex:worksAt ex:SomeVendor' is enough for RDFS to conclude it,")
    print("regardless of whether that is semantically sensible. An rdfs:domain/range")
    print("axiom is an inference TRIGGER (it manufactures a new fact), not a")
    print("validation CONSTRAINT (which would only ever report a problem, never add")
    print("a triple). SHACL is the right tool for the latter. See docs/ontology.md section 6.")


def demo_owl() -> None:
    """OWL-RL inference, including one case hard to express as property-graph traversal."""
    print("=== 1. owl:FunctionalProperty forcing owl:sameAs ===")
    print("tech:ownedBy is owl:FunctionalProperty. Fixture asserts TWO different")
    print("owners for the same service (inst:service-billing-api).")
    fixtures = load_store([paths.ONTOLOGY_V1, paths.REASONING_FIXTURES])
    materialize_owl_rl(fixtures)
    q = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX inst: <{INST}>
    SELECT ?other WHERE {{ inst:team-billing-ops owl:sameAs ?other }}
    """
    same_as = sorted({str(row.other) for row in fixtures.query(q)})
    print("inst:team-billing-ops owl:sameAs ->", same_as)
    print("Note inst:team-finance-platform is in that set: a Cypher query over the")
    print("identical two OWNED_BY edges would just see two ordinary, unrelated")
    print("edges. There is no Cypher-level mechanism that concludes two differently")
    print("-IRI'd nodes denote the same real-world entity from a cardinality rule")
    print("alone; that would need bespoke application code.")

    print()
    print("=== 2. owl:equivalentClass across heterogeneous systems ===")
    print("This is the headline 'hard to express as property-graph traversal' case")
    print("(see --demo interop for the full worked example):")
    print("sap:Client, crm:Customer, svcnow:BusinessPartner, and tech:Customer all")
    print("become mutually owl:equivalentClass once OWL-RL closure runs, so a query")
    print("for 'a tech:Customer' transparently matches individuals from all three")
    print("external systems with zero per-system relabeling ETL step.")

    print()
    print("=== 3. Disjoint-class consistency check ===")
    print("tech:Service owl:disjointWith tech:Team (among others). Fixture types one")
    print("individual under both.")
    violations = find_disjoint_class_violations(fixtures)
    for individual, class_a, class_b in violations:
        print(f"  VIOLATION: {individual} is typed under disjoint classes {class_a}, {class_b}")
    if not violations:
        print("  (no violations found)")


def _print_shacl_report(label: str, report) -> None:
    print(f"--- {label} ---")
    print(f"conforms={report.conforms}  violations={report.violation_count}")
    for line in report.violation_summaries:
        print(f"  {line}")
    print()


def demo_shacl() -> None:
    """SHACL validation, valid vs. invalid data, raw vs. RDFS-closed."""
    shapes = load_store([paths.SHAPES])

    valid_data = load_store(
        [paths.ONTOLOGY_V1, paths.INSTANCES_GRAPH_DEMO, paths.INSTANCES_CUSTOMER_DOMAIN]
    )
    _print_shacl_report(
        "A. Valid data, raw (inference=None)",
        run_shacl_validation(valid_data, shapes, inference=None),
    )
    _print_shacl_report(
        "A. Valid data, RDFS-closed (inference='rdfs')",
        run_shacl_validation(valid_data, shapes, inference="rdfs"),
    )
    print(
        "Both modes agree here (5 violations each, same entities): PostgreSQL, OpenSearch,\n"
        "Alpha Console Database, RAG Service, and Kafka all lack a tech:ownedBy triple in the\n"
        "real demo graph. This is a genuine data-quality finding, not a scripted result: the\n"
        "demo dataset was never given an owning team for its infrastructure dependencies.\n"
        "An earlier draft of this ontology's tech:dependsOn property had a narrow rdfs:domain\n"
        "declaration that made the raw and RDFS-closed reports DISAGREE (see KNOWN_ISSUES.md's\n"
        "ONT-001 entry and --demo domain-range-pitfall for that real, caught, and fixed bug).\n"
    )

    invalid_data = load_store(
        [paths.ONTOLOGY_V1, paths.INSTANCES_CUSTOMER_DOMAIN, paths.INSTANCES_INVALID]
    )
    _print_shacl_report(
        "B. Deliberately invalid data", run_shacl_validation(invalid_data, shapes, inference=None)
    )


def demo_sparql() -> None:
    """Run all 8 representative SPARQL queries, raw and reasoned, side by side."""
    raw_store = load_store(paths.ALL_VALID_FILES)
    reasoned_store = raw_store.copy()
    materialize_owl_rl(reasoned_store)

    query_files = sorted(glob.glob(str(paths.SPARQL_DIR / "*.rq")))
    for query_path in query_files:
        name = query_path.replace("\\", "/").rsplit("/", 1)[-1]
        raw_rows = run_query_file(raw_store, query_path)
        reasoned_rows = run_query_file(reasoned_store, query_path)
        print(f"=== {name} ===")
        print(f"raw: {len(raw_rows)} row(s)")
        print(format_rows(raw_rows))
        if reasoned_rows != raw_rows:
            print(f"after OWL-RL closure: {len(reasoned_rows)} row(s)")
            print(format_rows(reasoned_rows))
        else:
            print("(after OWL-RL closure: identical results)")
        print()


CYPHER_SPARQL_PAIRS = [
    ("01_services_owned_by_team.cypher", "01_services_owned_by_team.rq"),
    ("02_dependencies_of_service.cypher", "02_dependencies_of_service.rq"),
    ("03_policies_applying_to_service.cypher", "03_policies_applying_to_service.rq"),
    ("04_multi_hop_dependency_traversal.cypher", "04_multi_hop_dependency_traversal.rq"),
    ("05_runbooks_for_affected_services.cypher", "05_runbooks_for_affected_services.rq"),
]


def demo_property_graph_vs_rdf() -> None:
    """Show the same facts modeled twice, Cypher vs. SPARQL.

    Does NOT connect to a live Neo4j instance (none was running when
    this milestone was built). `rag.ontology.graphrag_integration`'s
    `expand_and_query_live_neo4j` is the function that does, when one
    is available). Prints the Cypher query text alongside the live
    SPARQL result from the matching file.
    """
    store = load_store(paths.ALL_VALID_FILES)

    for cypher_name, sparql_name in CYPHER_SPARQL_PAIRS:
        cypher_text = (paths.CYPHER_DIR / cypher_name).read_text(encoding="utf-8")
        sparql_rows = run_query_file(store, paths.SPARQL_DIR / sparql_name)

        print(f"=== {cypher_name}  <->  {sparql_name} ===")
        print("--- Cypher (property graph; text reproduced, not re-executed here) ---")
        print(cypher_text.strip())
        print("--- SPARQL (RDF mirror of the same facts; executed live, below) ---")
        print(format_rows(sparql_rows))
        print()

    note_path = paths.CYPHER_DIR / "06_query_subclasses_NOTE.txt"
    print("=== Where there is NO Cypher counterpart at all ===")
    print(Path(note_path).read_text(encoding="utf-8").strip())


def demo_interop() -> None:
    """Semantic interoperability and canonical entity resolution."""
    store = load_store(paths.ALL_VALID_FILES)

    print("=== Schema-level terminology mapping (owl:equivalentClass) ===")
    for mapping in find_equivalent_external_classes(store):
        print(f"  {_short(mapping.external_class)} -> {_short(mapping.canonical_class)}")

    print()
    print("=== Instance-level entity resolution ===")
    for link in summarize_entity_resolution(store):
        external_short = _short(link.external_entity)
        if link.confirmed:
            print(
                f"  CONFIRMED  canonical <-owl:sameAs-> {external_short} "
                f"(resolved_by={link.resolved_by})"
            )
        else:
            print(
                f"  PROPOSED   canonical <-tech:candidateSameAs-> {external_short} "
                f"(status={link.status}, resolved_by={link.resolved_by}); NOT merged"
            )
    print(
        "\n  (svcnow:organization_xyz appears in neither list above: deliberately left\n"
        "  unresolved, an identity conflict rather than a guessed merge; see\n"
        "  data/ontology/instances/canonical_entity_resolution.ttl.)"
    )

    print()
    print("=== The type-compatibility-is-not-identity subtlety ===")
    q = f"""
    PREFIX tech: <{TECH}>
    SELECT ?customer WHERE {{ ?customer a tech:Customer }}
    """
    reasoned = store.copy()
    materialize_owl_rl(reasoned)
    customers = sorted({_short(str(row.customer)) for row in reasoned.query(q)})
    print(f"  After OWL-RL closure, 'a tech:Customer' matches: {customers}")
    print(
        "  organization_xyz is in that list purely because its OWN asserted class\n"
        "  (svcnow:BusinessPartner) is schema-mapped to tech:Customer, NOT because its\n"
        "  identity was ever resolved to the canonical Northwind Logistics record (it\n"
        "  was not, see above). 'Type-compatible with Customer' and 'IS this specific\n"
        "  customer' are different claims this model deliberately keeps separate."
    )


def demo_auth() -> None:
    """Ontology reasoning never bypasses tenant/role authorization."""
    store = load_store(paths.ALL_VALID_FILES)
    materialize_owl_rl(store)  # inference runs BEFORE authorization, same order as real queries

    alpha_db = f"{INST}database-alpha-console-database"
    props = fetch_entity_auth_properties(store, alpha_db)
    print(f"Alpha Console Database auth properties: {props}")
    print()

    contexts = {
        "no identity (internal/unauthenticated)": None,
        "tenant_alpha + tenant_alpha_operator": AuthorizationContext(
            tenant_id="tenant_alpha", roles=["tenant_alpha_operator"]
        ),
        "tenant_beta + tenant_beta_operator": AuthorizationContext(
            tenant_id="tenant_beta", roles=["tenant_beta_operator"]
        ),
    }

    for label, auth in contexts.items():
        print(f"{label}: visible={is_entity_authorized(props, auth)}")

    cross_tenant_auth = AuthorizationContext(
        tenant_id="tenant_beta", roles=["tenant_alpha_operator"]
    )
    cross_tenant_visible = is_entity_authorized(
        props, cross_tenant_auth, cross_tenant_support_roles=["tenant_alpha_operator"]
    )
    print(
        "tenant_beta + a role LISTED on the entity's own allowed_roles "
        f"(cross-tenant support): visible={cross_tenant_visible}"
    )

    print()
    print("Every one of these checks ran AFTER OWL-RL closure (materialize_owl_rl above),")
    print("proving inference ran first and authorization second, never the other way. An")
    print("inferred fact about this entity (e.g. its entailed tech:Service membership via")
    print("tech:DatabaseService) changes nothing about who may see the entity itself: a")
    print("semantic relationship never grants data access on its own.")

    print()
    print("Cross-check: an UNSCOPED entity (no tech:tenantId triple at all) must remain")
    print("visible to every caller, inference or not:")
    rag_props = fetch_entity_auth_properties(store, f"{INST}service-rag-service")
    beta_auth = contexts["tenant_beta + tenant_beta_operator"]
    print(f"  RAG Service auth properties: {rag_props}")
    print(f"  tenant_beta caller sees it: {is_entity_authorized(rag_props, beta_auth)}")


DEMOS = {
    "rdfs": demo_rdfs,
    "domain-range-pitfall": demo_domain_range_pitfall,
    "owl": demo_owl,
    "shacl": demo_shacl,
    "sparql": demo_sparql,
    "property-graph-vs-rdf": demo_property_graph_vs_rdf,
    "interop": demo_interop,
    "auth": demo_auth,
}


def main() -> None:
    """CLI entrypoint: run one demo, or all of them in teaching order."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo", choices=[*DEMOS.keys(), "all"], default="all")
    args = parser.parse_args()

    names = list(DEMOS.keys()) if args.demo == "all" else [args.demo]
    for name in names:
        print("=" * 70)
        print(f"DEMO: {name}")
        print("=" * 70)
        DEMOS[name]()
        print()


if __name__ == "__main__":
    main()
