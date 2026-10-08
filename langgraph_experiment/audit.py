"""A small, isolated audit-event logger for this experiment's write-action workflow.

Deliberately not `rag.audit.log_audit_event`: that module's
`AuthEventType` is a fixed `Literal` shared by the production API/agent/
MCP code, and adding new event names to it for a learning experiment
would mean editing production source under `src/rag/`, exactly what
this package's isolation is meant to avoid (see `__init__.py`'s module
docstring). This module mirrors `rag.audit`'s shape and discipline
(structured JSON via the standard logger, IDs/enums/counts only, never
raw secrets) as its own, independent copy of the *pattern*, not the code.

Every event here answers the prompt spec's "safe workflow audit trail"
requirement: `thread_id`/`operation_id`/`case_id`/`new_status` are safe
identifiers; nothing here is ever a JWT, a raw approval payload, or
model-generated free text.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

_audit_logger = logging.getLogger("langgraph_experiment.audit")

WorkflowAuditEvent = Literal[
    "workflow_started",
    "write_action_requested",
    "write_action_invalid_request",
    "approval_granted",
    "approval_rejected",
    "approval_denied_insufficient_role",
    "approval_expired",
    "workflow_cancelled",
    "workflow_timed_out",
    "write_action_ledger_replay",
    "write_action_executed",
]


def log_workflow_event(event: WorkflowAuditEvent, **fields: Any) -> None:
    """Emit one structured audit log line for the experimental workflow.

    Parameters
    ----------
    event : WorkflowAuditEvent
        One of the fixed event names above.
    **fields : Any
        Structured fields: IDs, enum values, counts, timestamps only.
        Never a JWT, an approval payload's free-text `reason`, or
        anything chain-of-thought-shaped.
    """
    _audit_logger.info(event, extra=fields)
