"""Structured security-audit logging (auth-boundary milestone).

Lives at the top level (not under `api/`) because it's called from both
the API boundary (`api/deps.py`, `api/routers/*.py`) and retrieval-layer
code (`retrieval/pipeline.py`) -- `retrieval/` must never import from
`api/`, matching every other swap-point module in this codebase (see
CLAUDE.md's directory map). Reuses `logging_config.py`'s existing
`JSONFormatter` + request-id contextvar wholesale -- no new formatter/
handler/sink is introduced, only new `logger.info(...)` call sites on a
dedicated `rag.audit` logger name (filterable/routable independently of
`rag.api`'s general request logs later, e.g. to a different sink or an
OpenTelemetry exporter, without a schema change). Every event logs only
IDs/categories/counts already available on existing objects -- never JWT
contents, raw secrets, sensitive chunk text, or API keys, per the
milestone's explicit requirement. The JWT `sub` claim in particular is
never logged raw (its opacity/PII-safety is not guaranteed for this
milestone's tokens) -- `pseudonymous_subject` hashes it first.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

_audit_logger = logging.getLogger("rag.audit")

AuthEventType = Literal[
    "auth_success",
    "auth_failure",
    "authorization_denied",
    "cross_tenant_attempt",
    "field_redaction_applied",
    "trust_policy_rejection",
    "freshness_version_selected",
    "injection_flagged",
    "forged_claim_attempt",
    "rate_limit_exceeded",
    "oversized_request_rejected",
    "egress_policy_blocked",
]


def pseudonymous_subject(subject: str) -> str:
    """Return a stable, non-reversible identifier for a JWT `sub` claim.

    Parameters
    ----------
    subject : str
        The raw JWT `sub` claim value.

    Returns
    -------
    str
        A truncated sha256 hex digest of `subject` -- stable across calls
        for the same subject (so the same caller's events can be
        correlated), but never reversible to the original value.
    """
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]


def log_audit_event(event: AuthEventType, **fields: Any) -> None:
    """Emit one structured security-audit log record.

    Parameters
    ----------
    event : AuthEventType
        One of the fixed audit event names.
    **fields : Any
        Additional structured fields -- IDs, categories, counts only.
        Callers must never pass raw token contents, secrets, or chunk
        text here; see this module's docstring.
    """
    _audit_logger.info(event, extra=fields)
