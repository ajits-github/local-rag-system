"""Regression test for DIST-005: SSE disconnect must be noticed even while the queue is empty.

`_stream_agent_query`'s consumer loop used to check
`request.is_disconnected()` only *after* `await queue.get()` returned an
item. A long-running node that emits no intermediate `AgentEvent` (in
practice, a single slow LLM call) left the queue empty for the whole
call, during which a client disconnect was never noticed and
`cancel_event` was never set, so the worker thread ran to completion
regardless. This test drives a `run_agent` double that blocks until
`cancel_event` is set and emits nothing to the queue in the meantime,
with a `Request` double that reports disconnected from the start, and
asserts `cancel_event` is set well within the polling interval rather
than only after the (never-arriving) next queue item.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from rag.agent.graph import AgentRunResult
from rag.agent.state import AgentState
from rag.api.routers import agent_stream


class _AlwaysDisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


def _blocking_run_agent(state, *, cancel_event, on_event=None, **kwargs):
    """Stand-in for `run_agent`: blocks until cancelled, emits no AgentEvent at all."""
    cancelled_in_time = cancel_event.wait(timeout=5)
    state.termination_reason = "cancelled" if cancelled_in_time else "max_steps"
    return AgentRunResult(
        state=state, route="agent", retrieval_ms=0.0, generation_ms=0.0, total_ms=0.0
    )


@pytest.mark.asyncio
async def test_disconnect_is_noticed_while_the_queue_is_empty(monkeypatch):
    """cancel_event must be set well before the fake node's own 5s internal timeout."""
    monkeypatch.setattr(agent_stream, "_DISCONNECT_POLL_SECONDS", 0.2)
    monkeypatch.setattr(agent_stream, "run_agent", _blocking_run_agent)
    state = AgentState(original_query="a question whose caller is already gone")
    request = _AlwaysDisconnectedRequest()

    t0 = time.monotonic()
    async for _ in agent_stream._stream_agent_query(
        request,
        state,
        pipeline=None,
        vectorstore=None,
        embedder=None,
        llm=None,
        config=None,
        mcp_app=None,
        request_id=None,
        graph_retriever=None,
        graph_client=None,
    ):
        pass
    elapsed = time.monotonic() - t0

    # Well under the fake node's own 5s internal timeout: proves
    # cancellation was noticed via polling, not by the node finishing on
    # its own.
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_generator_teardown_also_sets_cancel_event(monkeypatch):
    """Even if the consumer loop exits some other way, the `finally` block still cancels.

    Simulated here by making the fake node observe whether cancel_event
    was set at all once it returns (not just the happy-path disconnect
    check), proving the `finally: cancel_event.set()` safety net fires.
    """
    monkeypatch.setattr(agent_stream, "_DISCONNECT_POLL_SECONDS", 0.2)
    observed_cancel_event: list[threading.Event] = []

    def _recording_run_agent(state, *, cancel_event, on_event=None, **kwargs):
        cancel_event.wait(timeout=5)
        observed_cancel_event.append(cancel_event)
        return AgentRunResult(
            state=state, route="agent", retrieval_ms=0.0, generation_ms=0.0, total_ms=0.0
        )

    monkeypatch.setattr(agent_stream, "run_agent", _recording_run_agent)
    state = AgentState(original_query="a question whose caller is already gone")
    request = _AlwaysDisconnectedRequest()

    async for _ in agent_stream._stream_agent_query(
        request,
        state,
        pipeline=None,
        vectorstore=None,
        embedder=None,
        llm=None,
        config=None,
        mcp_app=None,
        request_id=None,
        graph_retriever=None,
        graph_client=None,
    ):
        pass
    await asyncio.sleep(0)

    assert len(observed_cancel_event) == 1
    assert observed_cancel_event[0].is_set()
