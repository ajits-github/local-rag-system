"""Demo caller identities: plain tenant/role claims, never a real JWT.

This experiment is about LangGraph graph mechanics, not re-proving the
JWT authentication boundary `rag.api.auth`/`rag.api.deps` already covers
(and which the production custom harness sits behind). A `DemoIdentity`
is asserted directly by the CLI caller, mirroring how
`eval/run_eval.py`'s gold-driven harness and `security.auth.enabled=False`
mode already treat caller-supplied tenant/roles as trusted claims in this
codebase -- not a new, weaker convention invented for this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.api.auth import VerifiedIdentity
from rag.retrieval.authorization import AuthorizationContext


@dataclass(frozen=True)
class DemoIdentity:
    """One demo caller's identity claims, for both retrieval and case-store authorization."""

    subject: str = "demo-caller"
    tenant_id: str | None = None
    roles: tuple[str, ...] = field(default_factory=tuple)

    def to_verified_identity(self) -> VerifiedIdentity:
        """Build the `VerifiedIdentity` shape `rag.mcp.business.store` expects."""
        return VerifiedIdentity(
            subject=self.subject,
            tenant_id=self.tenant_id,
            roles=list(self.roles),
            issuer=None,
            audience=None,
        )

    def to_authorization_context(self) -> AuthorizationContext:
        """Build the `AuthorizationContext` shape `RetrievalPipeline` expects."""
        return AuthorizationContext(tenant_id=self.tenant_id, roles=list(self.roles))
