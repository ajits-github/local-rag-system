from __future__ import annotations

import logging

from rag.audit import log_audit_event, pseudonymous_subject


def test_pseudonymous_subject_is_stable_and_non_reversible():
    """The same subject always hashes to the same identifier, and it isn't the raw value."""
    first = pseudonymous_subject("alice@example.com")
    second = pseudonymous_subject("alice@example.com")

    assert first == second
    assert first != "alice@example.com"
    assert "alice" not in first


def test_pseudonymous_subject_differs_for_different_subjects():
    """Two different subjects hash to different identifiers (no accidental collision)."""
    assert pseudonymous_subject("alice") != pseudonymous_subject("bob")


def test_log_audit_event_emits_one_record_with_event_as_message(caplog):
    """log_audit_event() logs exactly one record whose message is the event name."""
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        log_audit_event("auth_success", subject="abc123", tenant_id="tenant_alpha")

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "auth_success"


def test_log_audit_event_merges_extra_fields_onto_the_record(caplog):
    """Fields passed as **fields land as attributes on the emitted LogRecord."""
    with caplog.at_level(logging.INFO, logger="rag.audit"):
        log_audit_event(
            "field_redaction_applied", field_ids=["synthetic_admin_credential"], count=2
        )

    record = caplog.records[0]
    assert record.field_ids == ["synthetic_admin_credential"]
    assert record.count == 2


def test_auth_failure_event_never_contains_raw_token(caplog):
    """An auth_failure event logs only a reason category, never the raw JWT string."""
    raw_token = "eyJhbGciOiJIUzI1NiJ9.super-secret-payload.signature"

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        log_audit_event("auth_failure", reason="invalid_signature")

    record = caplog.records[0]
    record_text = str(vars(record))
    assert raw_token not in record_text
    assert "token" not in vars(record)
    assert "raw_token" not in vars(record)


def test_field_redaction_event_never_contains_raw_chunk_content(caplog):
    """A field_redaction_applied event logs field_ids/counts only, never chunk content."""
    raw_secret_value = "SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE"

    with caplog.at_level(logging.INFO, logger="rag.audit"):
        log_audit_event(
            "field_redaction_applied",
            field_ids=["synthetic_admin_credential"],
            redacted_chunk_count=1,
        )

    record = caplog.records[0]
    record_text = str(vars(record))
    assert raw_secret_value not in record_text
    assert "content" not in vars(record)
