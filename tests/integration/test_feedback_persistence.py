"""Real-Postgres round-trip tests for FeedbackStore.

Assumes `make up` (which now also runs `scripts/init_db.py`'s
`build_feedback_schema_sql`) has already created the `feedback` table,
matching every other integration test's "infra is already up" assumption
(see `conftest.py`'s `require_postgres`). Self-skips when Postgres isn't
reachable.
"""

from __future__ import annotations

import uuid

from rag.feedback.store import FeedbackStore, FeedbackWriteInput


def _store(config) -> FeedbackStore:
    return FeedbackStore(dsn=config.database_url(), table=config.feedback.table_name)


def _write_input(**overrides) -> FeedbackWriteInput:
    base = dict(
        tenant_id=None,
        caller_key="pytest-caller",
        request_id=str(uuid.uuid4()),
        rating="positive",
        reason=None,
        comment=None,
        route="classic_rag",
        dataset_id="pytest-integration",
        query_text="what is the retention policy?",
        answer_text="90 days.",
        cited_source_ids=["doc_0"],
        tool_calls=[],
        generation_model="qwen2.5:1.5b",
        prompt_id="rag_answer",
        prompt_version="v1",
        retrieval_provider="dense",
        reranker_provider="none",
    )
    base.update(overrides)
    return FeedbackWriteInput(**base)


def test_submit_persists_full_row_contents(require_postgres, config):
    """A submitted row round-trips every field, including lineage, through export()."""
    store = _store(config)
    write_input = _write_input(tenant_id="tenant_alpha", caller_key=f"pytest-{uuid.uuid4()}")

    feedback_id, created = store.submit(write_input)
    try:
        assert created is True
        rows = store.export(tenant_id="tenant_alpha", limit=1000)
        row = next(r for r in rows if r.feedback_id == feedback_id)

        assert row.tenant_id == "tenant_alpha"
        assert row.caller_key == write_input.caller_key
        assert row.request_id == write_input.request_id
        assert row.rating == "positive"
        assert row.route == "classic_rag"
        assert row.dataset_id == "pytest-integration"
        assert row.query_text == "what is the retention policy?"
        assert row.answer_text == "90 days."
        assert row.cited_source_ids == ["doc_0"]
        assert row.generation_model == "qwen2.5:1.5b"
        assert row.prompt_id == "rag_answer"
        assert row.created_at is not None
        assert row.updated_at is not None
    finally:
        store.delete(feedback_id)


def test_second_submission_by_same_caller_and_run_updates_the_row(require_postgres, config):
    """Same (tenant, caller_key, request_id) twice updates one row rather than creating a second."""
    store = _store(config)
    caller_key = f"pytest-{uuid.uuid4()}"
    request_id = str(uuid.uuid4())

    first_id, first_created = store.submit(
        _write_input(
            tenant_id="tenant_alpha",
            caller_key=caller_key,
            request_id=request_id,
            rating="positive",
        )
    )
    try:
        second_id, second_created = store.submit(
            _write_input(
                tenant_id="tenant_alpha",
                caller_key=caller_key,
                request_id=request_id,
                rating="negative",
                reason="incorrect_answer",
            )
        )

        assert first_created is True
        assert second_created is False
        assert first_id == second_id

        rows = store.export(tenant_id="tenant_alpha", limit=1000)
        matching = [r for r in rows if r.feedback_id == first_id]
        assert len(matching) == 1
        assert matching[0].rating == "negative"
        assert matching[0].reason == "incorrect_answer"
        assert matching[0].updated_at >= matching[0].created_at
    finally:
        store.delete(first_id)


def test_two_anonymous_no_tenant_submissions_do_not_collide(require_postgres, config):
    """Two distinct request_ids under the same (empty tenant, caller_key) each get their own row.

    Exercises the '' tenant-id sentinel specifically: distinct request_ids
    must not be conflated just because neither submission asserted a
    tenant.
    """
    store = _store(config)
    caller_key = f"pytest-anon-{uuid.uuid4()}"

    id_a, created_a = store.submit(
        _write_input(tenant_id=None, caller_key=caller_key, request_id=str(uuid.uuid4()))
    )
    id_b, created_b = store.submit(
        _write_input(tenant_id=None, caller_key=caller_key, request_id=str(uuid.uuid4()))
    )

    try:
        assert created_a is True
        assert created_b is True
        assert id_a != id_b
    finally:
        store.delete(id_a)
        store.delete(id_b)


def test_export_filters_by_rating_and_dataset_id(require_postgres, config):
    """export() only returns rows matching every supplied filter."""
    store = _store(config)
    dataset_id = f"pytest-export-{uuid.uuid4()}"
    positive_id, _ = store.submit(
        _write_input(dataset_id=dataset_id, rating="positive", caller_key=f"pytest-{uuid.uuid4()}")
    )
    negative_id, _ = store.submit(
        _write_input(dataset_id=dataset_id, rating="negative", caller_key=f"pytest-{uuid.uuid4()}")
    )

    try:
        negative_rows = store.export(dataset_id=dataset_id, rating="negative")
        assert {r.feedback_id for r in negative_rows} == {negative_id}

        all_rows = store.export(dataset_id=dataset_id)
        assert {r.feedback_id for r in all_rows} == {positive_id, negative_id}
    finally:
        store.delete(positive_id)
        store.delete(negative_id)
