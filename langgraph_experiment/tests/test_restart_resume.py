"""State survives a real checkpoint reload across a simulated process restart.

`InMemorySaver` (every other test file) cannot demonstrate this by
construction -- it's only ever one Python process's memory. This file
uses the real `SqliteSaver` against a `tmp_path` file, opening and
closing two independent connections to prove the second one -- standing
in for a freshly started process, exactly like `run_demo.py`/
`approval_cli.py` each opening their own connection per invocation --
picks up precisely where the first left off.
"""

from __future__ import annotations

from langgraph.types import Command

from langgraph_experiment.checkpointer import sqlite_checkpointer
from langgraph_experiment.graph import build_graph
from rag.mcp.business import store as case_store


def test_paused_state_reloads_after_reopening_the_checkpoint_db(deps, tmp_path, thread_config):
    db_path = tmp_path / "restart_test.sqlite"
    write_state = {
        "original_query": "Please close CASE-2001",
        "tenant_id": "tenant_beta",
        "roles": ["tenant_beta_operator"],
    }

    # "Process 1": start the run, reach the interrupt, then close the connection.
    with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_graph(deps, checkpointer)
        graph.invoke(write_state, thread_config)
        snapshot = graph.get_state(thread_config)
        assert snapshot.next == ("wait_for_approval",)

    # No mutation happened before the "restart".
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "resolved"

    # "Process 2": a brand-new connection to the same file, no shared Python
    # objects with process 1 at all beyond the db path itself.
    with sqlite_checkpointer(db_path) as checkpointer:
        graph2 = build_graph(deps, checkpointer)
        reloaded = graph2.get_state(thread_config)
        assert reloaded.next == ("wait_for_approval",)
        assert reloaded.interrupts[0].value["case_id"] == "CASE-2001"

        result = graph2.invoke(
            Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
            thread_config,
        )
        assert result["termination_reason"] == "executed"

    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"


def test_full_start_stop_restart_approve_resume_complete_sequence(deps, tmp_path, thread_config):
    """The exact 8-step sequence from the prompt spec's "Restart/resume experiment" section."""
    db_path = tmp_path / "sequence_test.sqlite"

    # 1. start workflow
    with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_graph(deps, checkpointer)
        graph.invoke(
            {
                "original_query": "Please close CASE-2001",
                "tenant_id": "tenant_beta",
                "roles": ["tenant_beta_operator"],
            },
            thread_config,
        )
        # 2. reach approval interrupt
        assert graph.get_state(thread_config).next == ("wait_for_approval",)
    # 3. stop the process (context manager exit closes the sqlite connection)

    # 4. restart it / 5. load the existing checkpoint
    with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_graph(deps, checkpointer)
        assert graph.get_state(thread_config).next == ("wait_for_approval",)

        # 6. approve / 7. resume
        result = graph.invoke(
            Command(resume={"decision": "approve", "approver_roles": ["case_status_approver"]}),
            thread_config,
        )

    # 8. complete action
    assert result["termination_reason"] == "executed"
    assert case_store._SYNTHETIC_CASES["CASE-2001"].status == "closed"
