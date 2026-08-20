"""Structured security-audit logging.

Lives at the top level because it is called from both the API boundary
(`api/deps.py`, `api/routers/*.py`) and retrieval-layer code. Keeping it
outside `api/` preserves the API boundary and retrieval-layer separation.
It reuses `logging_config.py`'s existing `JSONFormatter` and request-id
context variable.

Notes
-----
Every event logs only IDs, categories, and counts already available on
existing objects. JWT contents, raw secrets, sensitive chunk text, and API
keys must never be logged. The JWT `sub` claim is hashed by
`pseudonymous_subject` before logging because its opacity is not
guaranteed.
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
    "agent_tool_argument_rejected",
    "agent_max_steps_reached",
    "agent_max_retrieval_attempts_reached",
    "agent_max_tool_calls_reached",
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
        A truncated sha256 hex digest of `subject`. Stable across calls
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
        Additional structured fields. Callers must pass only IDs,
        categories, and counts, never raw token contents, secrets, or
        chunk text.
    """
    _audit_logger.info(event, extra=fields)
