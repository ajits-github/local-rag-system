"""JWT verification at the API boundary (auth-boundary milestone).

Authentication ("who is the caller?") lives entirely here. Authorization
("what may this verified caller access?") stays in
`retrieval/authorization.py`/`retrieval/pipeline.py`/`vectorstore/pgvector.py`,
unchanged -- this module never constructs an `AuthorizationContext` and
never imports `RetrievalPipeline`/`VectorStore`. `VerifiedIdentity` is the
one-way handoff between the two: `api/routers/query.py` maps its
`tenant_id`/`roles` onto `AuthorizationContext`, nothing more.
"""

from __future__ import annotations

from typing import Literal

import jwt
from pydantic import BaseModel, Field

from rag.config import AppConfig

AuthFailureReason = Literal[
    "missing_token",
    "invalid_signature",
    "expired",
    "malformed",
    "missing_claim",
    "invalid_issuer",
    "invalid_audience",
]


class AuthenticationError(Exception):
    """A JWT failed verification.

    Carries a `reason` (never the token itself, never a raw exception
    message that might embed claim values) so callers can log a precise,
    audit-safe failure category.
    """

    def __init__(self, reason: AuthFailureReason, detail: str = "") -> None:
        """Store the failure `reason` for audit logging and the 401 response."""
        self.reason = reason
        super().__init__(detail or reason)


class VerifiedIdentity(BaseModel):
    """Claims extracted from a verified JWT.

    `subject`/`issuer`/`audience` are never mapped onto
    `AuthorizationContext` -- that model stays retrieval-scoped. They're
    used only for audit logging at the API boundary (`subject` always via
    a hashed/pseudonymous form, see `api/audit.py`), then discarded.
    """

    subject: str
    tenant_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    issuer: str | None = None
    audience: str | None = None


def verify_jwt(token: str, config: AppConfig) -> VerifiedIdentity:
    """Verify `token` per `config.security.auth.jwt` and return its identity.

    Verifies signature, expiration, issuer (if configured), audience (if
    configured), and presence of every claim in
    `config.security.auth.jwt.required_claims` -- rejecting a token
    missing any of them as malformed, per requirement 2's "reject ...
    token missing required tenant/role claims."

    Parameters
    ----------
    token : str
        The raw JWT string (without a leading ``"Bearer "``).
    config : AppConfig
        Application configuration; reads `security.auth.jwt` and resolves
        the signing key via `config.jwt_signing_key()`.

    Returns
    -------
    VerifiedIdentity
        The caller's verified identity claims.

    Raises
    ------
    AuthenticationError
        On any verification failure -- signature, expiration, issuer,
        audience, or a missing required claim. The token's contents are
        never included in the raised exception.
    """
    jwt_config = config.security.auth.jwt
    key = config.jwt_signing_key()
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[jwt_config.algorithm],
            issuer=jwt_config.issuer,
            audience=jwt_config.audience,
            leeway=jwt_config.leeway_seconds,
            options={"require": list(jwt_config.required_claims)},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("expired") from exc
    except jwt.InvalidSignatureError as exc:
        raise AuthenticationError("invalid_signature") from exc
    except jwt.InvalidIssuerError as exc:
        raise AuthenticationError("invalid_issuer") from exc
    except jwt.InvalidAudienceError as exc:
        raise AuthenticationError("invalid_audience") from exc
    except jwt.MissingRequiredClaimError as exc:
        raise AuthenticationError("missing_claim") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("malformed") from exc

    return VerifiedIdentity(
        subject=str(claims["sub"]),
        tenant_id=claims.get("tenant_id"),
        roles=list(claims.get("roles") or []),
        issuer=claims.get("iss"),
        audience=claims.get("aud"),
    )
