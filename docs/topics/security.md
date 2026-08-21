# Security

--8<-- "README.md:docs-security"

Each layer above is independently toggleable (`config.security.*`, every
flag defaults to off/no-op):

- **Authorization, freshness, and trust**: tenant/role SQL predicates run
  in Postgres before a row ever leaves the database; document-version
  freshness resolution picks the currently-effective member of a
  superseded-document family; trust level is stored and filterable, not a
  hard gate by default.
- **Field-level redaction**: a document a caller may legitimately retrieve
  can still contain one specific field (e.g. a credential) that most
  readers of that document must not see verbatim. Redaction runs on
  retrieved chunk text and metadata before the prompt is built, so an
  unauthorized value structurally cannot reach the generation model.
- **Authenticated API boundary**: JWT verification at the API boundary
  produces a verified identity; authorization stays enforced entirely at
  retrieval. The two are structurally separate modules by design.
- **Prompt-injection detection** is telemetry/reinforcement, not a gate:
  authorization is the actual control.

Full design writeups:

- [Authorization, Freshness, and Trust](../architecture.md#authorization-freshness-and-trust-safetyfreshness-milestone)
- [Field-Level Sensitive-Data Redaction](../architecture.md#field-level-sensitive-data-redaction-field-level-safety-milestone)
- [Authenticated API Boundary and Security Hardening](../architecture.md#authenticated-api-boundary-and-security-hardening-auth-boundary-milestone)
- API reference: [Retrieval](../reference/retrieval.md), [API](../reference/api.md)
