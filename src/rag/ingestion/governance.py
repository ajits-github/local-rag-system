"""Ingestion-time caller-identity governance: resolves a newly-ingested document's tenant.

Closes the gap where an authenticated upload with no (or mismatched)
governance metadata could silently become `tenant_id=NULL` (globally
visible to every tenant) or be assigned to an arbitrary tenant. Document
level authorization (`retrieval/authorization.py`) already enforces tenant
scoping at query time; this module is what makes sure a chunk actually
carries the *right* `tenant_id` in the first place for documents ingested
through the authenticated `POST /ingest` boundary.

Deliberately decoupled from `rag.api.auth.VerifiedIdentity`: `rag.ingestion`
must never import from `rag.api`, the same API-boundary/pipeline-layer
separation `rag.retrieval`/`rag.api.auth` already enforce (see `CLAUDE.md`'s
"Authenticated API boundary" section). `IngestCallerContext` carries only the
two primitive facts this module needs, built by the API router from a
`VerifiedIdentity` and passed in as plain data.
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestCallerContext(BaseModel):
    """The two governance-relevant facts about an authenticated ingest caller.

    Parameters
    ----------
    tenant_id : str or None
        The caller's own verified tenant, from `VerifiedIdentity.tenant_id`.
        `None` only if the deployment's JWT config doesn't require a
        `tenant_id` claim (`security.auth.jwt.required_claims`); a
        tenant-less identity can still ingest, but every document it
        ingests without an explicit, privileged override also ends up
        `tenant_id=None`, the same "no claim to stamp from" limitation
        this fix cannot invent an answer for.
    is_privileged : bool
        Whether the caller holds a role in
        `security.authorization.cross_tenant_support_roles`, the same
        role list retrieval-time cross-tenant access already uses (see
        `vectorstore.pgvector.build_authorization_where_clause`). Reused
        deliberately rather than adding a second, parallel privilege list.
    """

    tenant_id: str | None = None
    is_privileged: bool = False


class IngestGovernanceError(Exception):
    """Raised when an ingest caller's requested tenant_id conflicts with their identity.

    Caught at the API boundary (`api/routers/ingest.py`) and turned into a
    403. Never raised for a `caller=None` call (unauthenticated ingestion --
    JWT auth disabled, or `insecure_dev_mode`) and the CLI/`make ingest`
    path, neither of which this module is ever invoked for).
    """


def resolve_ingest_tenant_id(
    parsed_tenant_id: str | None, caller: IngestCallerContext
) -> str | None:
    """Resolve the `tenant_id` a newly-ingested document should be persisted under.

    Precedence, given an authenticated `caller`:

    1. `parsed_tenant_id` missing (no governance front matter at all, or a
       PDF/DOCX/HTML upload; those loaders never produce one) -> the
       caller's own `tenant_id`. A document is never silently left
       untenanted/globally-visible just because the caller didn't author
       front matter.
    2. `parsed_tenant_id == caller.tenant_id` -> unchanged (the common,
       explicit self-tenant-scoped case).
    3. `parsed_tenant_id` names a *different* tenant:
       - `caller.is_privileged` -> honored as specified (a support/system
         role intentionally authoring content on another tenant's behalf).
       - otherwise -> rejected (`IngestGovernanceError`); a normal
         tenant-scoped caller can never assign their upload to a tenant
         other than their own by editing front matter.

    `allowed_roles`/`classification`/`trust_level`/etc. are untouched.
    this function governs `tenant_id` only, the one field whose absence the
    pgvector authorization predicate treats as globally visible
    (`tenant_id IS NULL`). A same-tenant, role-unrestricted default
    (`allowed_roles=None`) after correct tenant scoping is normal, expected
    behavior, not a gap this fix needs to close.

    Parameters
    ----------
    parsed_tenant_id : str | None
        `RawDocument.tenant_id` as the loader produced it (from front
        matter, or `None`).
    caller : IngestCallerContext
        The authenticated caller's governance-relevant identity facts.

    Returns
    -------
    str | None
        The `tenant_id` to persist onto the document/its chunks. Can still
        be `None` if `caller.tenant_id` itself is `None` (a tenant-less
        verified identity) and `parsed_tenant_id` was also missing,
        a documented, narrow edge case, not a silent default.

    Raises
    ------
    IngestGovernanceError
        If `parsed_tenant_id` names a different tenant than
        `caller.tenant_id` and `caller.is_privileged` is `False`.
    """
    if parsed_tenant_id is None:
        return caller.tenant_id
    if parsed_tenant_id == caller.tenant_id:
        return parsed_tenant_id
    if caller.is_privileged:
        return parsed_tenant_id
    raise IngestGovernanceError(
        f"Document specifies tenant_id={parsed_tenant_id!r}, which does not match "
        "the authenticated caller's own tenant, and the caller does not hold a "
        "cross-tenant support role."
    )
