"""Safe, structured live-progress events for a running agent query.

`AgentEvent` deliberately has no free-text field at all. Not
`reasoning`, not a message string, not evidence content. This is a
structural guarantee, not a policy one: there is nowhere on the model for
chain-of-thought, raw prompts, retrieved chunk text, or credentials to
end up, so no future call site can accidentally leak them into a stream
just by passing more data through. See `docs/architecture.md`'s
"Observability" section for the full design.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EventType = Literal[
    "query_received",
    "route_selected",
    "decomposition_started",
    "decomposition_completed",
    "tool_selected",
    "tool_started",
    "tool_completed",
    "evidence_evaluated",
    "retry_started",
    "synthesis_started",
    "completed",
    "terminated",
]


class AgentEvent(BaseModel):
    """One safe, operational event describing agent-graph progress.

    Attributes
    ----------
    event_type : EventType
        Which state-machine transition this event describes.
    step : int | None
        `AgentState.step_count` at the time of this event, if applicable.
    tool_name : str | None
        The bounded tool name (one of the four literal tools), if this
        event concerns a tool dispatch.
    elapsed_ms : float | None
        Milliseconds elapsed since the run started, if applicable.
    retrieved_chunk_count : int | None
        Count only. Never chunk ids or content.
    evidence_sufficient : bool | None
        `evaluate_evidence`'s outcome, if this event concerns it.
    termination_reason : str | None
        Set only on a `terminated`/`completed` event.
    route : str | None
        `"classic_rag"` or `"agent"`, set only on `route_selected`/
        `completed`/`terminated`.
    """

    event_type: EventType
    step: int | None = None
    tool_name: str | None = None
    elapsed_ms: float | None = None
    retrieved_chunk_count: int | None = None
    evidence_sufficient: bool | None = None
    termination_reason: str | None = None
    route: str | None = None
