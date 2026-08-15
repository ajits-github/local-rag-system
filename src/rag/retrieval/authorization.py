"""Retrieval-time authorization context.

`AuthorizationContext` is passed explicitly into `VectorStore` and
`RetrievalPipeline` methods. It is never inferred from caller-supplied
metadata filters, and this module does not verify JWTs or authenticate
callers.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AuthorizationContext(BaseModel):
    """Caller identity and date context for one retrieval, or `None` for unrestricted access.

    Parameters
    ----------
    tenant_id : str or None
        Tenant the caller belongs to. Chunks with no `tenant_id` are
        never gated, so passing a context is always additive.
    roles : list[str]
        Verified caller roles used for role-based authorization.
    as_of : date or None
        Date used for document-version freshness resolution (see
        `retrieval/freshness.py`). `None` means "current".
    include_superseded : bool
        When `True` and `as_of` is `None`, disables freshness filtering
        entirely.
    require_trust_level : str or None
        Required document trust level, if any.

    Notes
    -----
    This class represents authorization context only. Authentication is
    handled separately at the API boundary.
    """

    tenant_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    as_of: date | None = None
    include_superseded: bool = False
    require_trust_level: str | None = None
    # Populated by RetrievalPipeline after freshness resolution; callers
    # should not set this field directly.
    resolved_excluded_document_ids: list[str] = Field(default_factory=list)
