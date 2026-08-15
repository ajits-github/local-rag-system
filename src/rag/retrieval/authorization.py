"""Retrieval-time authorization context: who is asking, and as of when.

`AuthorizationContext` is passed *explicitly* into `VectorStore`/
`RetrievalPipeline` methods, never inferred from a caller-suppliable
`filters` dict (see `vectorstore.base.ALLOWED_FILTER_FIELDS`, which stays a
separate, still-caller-controllable exact-match mechanism for convenience
filters like `category`). This module has no dependency on
`rag.vectorstore`/`rag.ingestion`, so it's safe for the vectorstore layer to
import it without a cycle.

No authentication/session layer exists anywhere in this codebase (no JWT,
no login, no user model) -- `tenant_id`/`roles` are trusted claims the
caller (an API request, or `eval.run_eval`'s gold-driven harness) asserts
directly. This milestone implements *enforcement* given an already-asserted
identity, not identity issuance/verification; a real deployment would
populate `AuthorizationContext` from a verified JWT/session claim at the API
boundary instead of a raw request field. Documented here rather than
silently implied, per this milestone's "don't fabricate security guarantees
that aren't implemented" instruction.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AuthorizationContext(BaseModel):
    """Caller identity/date context for one retrieval, or `None` for unrestricted access.

    `tenant_id`/`roles` gate tenant- and role-scoped chunks (see
    `vectorstore.pgvector._build_authorization_where_clause`) -- chunks with
    `tenant_id IS NULL` (the entire pre-existing, pre-milestone corpus) are
    never gated regardless of this context, so passing an
    `AuthorizationContext` is strictly additive and never regresses
    retrieval over untenanted content.

    `as_of`/`include_superseded` drive freshness resolution (see
    `retrieval/freshness.py`): `as_of=None` (the default) means "current" --
    prefer/require `status="active"` members of a document-version family;
    an explicit `as_of` date deterministically resolves each family to the
    single version that was effective on that date (or excludes the whole
    family if none qualify -- see `freshness.py`'s documented limitations).
    `include_superseded=True` with `as_of=None` disables freshness filtering
    entirely (every version of every family stays eligible) -- a rarely
    -needed escape hatch, not exercised by the current gold set (which
    always pairs a historical question with an explicit `query_as_of`).
    """

    tenant_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    as_of: date | None = None
    include_superseded: bool = False
    # Section 7 ("knowledge-source trust"): when set, only chunks with
    # trust_level IS NULL (untenanted legacy content, never gated) or
    # trust_level == this value are visible -- e.g. "authoritative", so an
    # untrusted/poisoned-but-otherwise-authorized document (still tenant/
    # role-permitted -- see docs/architecture.md) cannot satisfy a query
    # that policy says requires an authoritative source. None (the
    # default) applies no trust filtering at all.
    require_trust_level: str | None = None
    # Populated by RetrievalPipeline.retrieve() from
    # retrieval/freshness.py's resolve_excluded_document_ids -- never set
    # directly by a caller-constructed context (there is no `document_id`
    # to know ahead of a query). Kept on this same context object purely so
    # VectorStore's search/search_keyword/get_chunks_by_* methods only need
    # one `auth` parameter, not a second one threaded everywhere in parallel.
    resolved_excluded_document_ids: list[str] = Field(default_factory=list)
