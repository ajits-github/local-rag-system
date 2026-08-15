from __future__ import annotations

from datetime import UTC, datetime

from rag.retrieval.field_policy import (
    SensitiveFieldPolicy,
    detect_sensitive_field_ids,
    find_duplicate_sensitive_occurrences,
    is_role_authorized_for_field,
    redact_sensitive_fields,
    redact_source_metadata,
)
from rag.schemas import Chunk, ChunkMetadata

_ALPHA_ADMIN_KEY = "SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE"


def _chunk(chunk_id: str, content: str, sensitive_field_ids: list[str] | None = None) -> Chunk:
    """Build a Chunk with minimal-but-valid metadata for duplicate-detection tests."""
    now = datetime.now(UTC)
    metadata = ChunkMetadata(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        source_type="text",
        created_at=now,
        last_modified=now,
        chunk_index=0,
        dataset_id="test-dataset",
        sensitive_field_ids=sensitive_field_ids,
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata)


def test_detect_sensitive_field_ids_matches_default_credential_policy():
    """A chunk containing the admin-key literal is tagged with its field_id."""
    text = f"The synthetic test key is `{_ALPHA_ADMIN_KEY}`."
    assert detect_sensitive_field_ids(text) == ["synthetic_admin_credential"]


def test_detect_sensitive_field_ids_empty_for_ordinary_text():
    """Plain operational text with no matching pattern gets no tags."""
    assert detect_sensitive_field_ids("The callback processor waits 45 seconds.") == []


def test_redact_sensitive_fields_redacts_for_unauthorized_role():
    """A role not in the policy's allowed_roles gets the value replaced with the marker."""
    text = f"The synthetic test key is `{_ALPHA_ADMIN_KEY}`."
    redacted, field_ids = redact_sensitive_fields(text, roles=["tenant_alpha_operator"])
    assert _ALPHA_ADMIN_KEY not in redacted
    assert "[REDACTED:SENSITIVE_FIELD]" in redacted
    assert field_ids == ["synthetic_admin_credential"]


def test_redact_sensitive_fields_preserves_for_authorized_role():
    """A role in the policy's allowed_roles sees the raw value unchanged."""
    text = f"The synthetic test key is `{_ALPHA_ADMIN_KEY}`."
    redacted, field_ids = redact_sensitive_fields(text, roles=["tenant_alpha_admin"])
    assert redacted == text
    assert field_ids == []


def test_redact_sensitive_fields_fails_closed_when_no_roles():
    """Regression test: a missing/empty identity must redact, never pass through unrestricted.

    `roles=[]` (no asserted identity) must never be treated as
    "unrestricted access" -- see FieldRedactionConfig's fail-closed
    requirement. `any(role in allowed_roles for role in [])` is False for
    every policy, so this asserts that behavior explicitly rather than
    relying on it being an accidental consequence of `any()`'s semantics.
    """
    text = f"The synthetic test key is `{_ALPHA_ADMIN_KEY}`."
    redacted, field_ids = redact_sensitive_fields(text, roles=[])
    assert _ALPHA_ADMIN_KEY not in redacted
    assert field_ids == ["synthetic_admin_credential"]


def test_redact_sensitive_fields_leaves_unrelated_text_untouched():
    """Text with no matching policy pattern is returned unchanged, no field_ids reported."""
    text = "The callback route ends in /v2 and the retry delay is 45 seconds."
    redacted, field_ids = redact_sensitive_fields(text, roles=[])
    assert redacted == text
    assert field_ids == []


def test_redact_sensitive_fields_replaces_every_match():
    """Multiple occurrences of a matched pattern are all replaced."""
    text = f"Key A: {_ALPHA_ADMIN_KEY}. Key A again: {_ALPHA_ADMIN_KEY}."
    redacted, field_ids = redact_sensitive_fields(text, roles=[])
    assert _ALPHA_ADMIN_KEY not in redacted
    assert redacted.count("[REDACTED:SENSITIVE_FIELD]") == 2
    assert field_ids == ["synthetic_admin_credential"]


def test_is_role_authorized_for_field_true_for_allowed_role():
    """A role present in the policy's allowed_roles is authorized."""
    assert is_role_authorized_for_field("synthetic_admin_credential", ["tenant_beta_admin"])


def test_is_role_authorized_for_field_false_for_disallowed_role():
    """A role absent from the policy's allowed_roles is not authorized."""
    assert not is_role_authorized_for_field("synthetic_admin_credential", ["tenant_alpha_operator"])


def test_is_role_authorized_for_field_false_for_no_roles():
    """No roles at all is never authorized for a known field."""
    assert not is_role_authorized_for_field("synthetic_admin_credential", [])


def test_is_role_authorized_for_field_true_for_unknown_field_id():
    """An id not backed by any policy has nothing to restrict -- returns True."""
    assert is_role_authorized_for_field("not_a_real_policy", [])


def test_custom_policies_override_defaults():
    """Callers can pass an explicit policy list instead of DEFAULT_FIELD_POLICIES."""
    policies = [
        SensitiveFieldPolicy(
            field_id="custom_secret",
            sensitivity_type="credential",
            pattern=r"CUSTOM_SECRET_\w+",
            allowed_roles=["custom_admin"],
        )
    ]
    text = "The value is CUSTOM_SECRET_ABC123."
    redacted, field_ids = redact_sensitive_fields(text, roles=[], policies=policies)
    assert "CUSTOM_SECRET_ABC123" not in redacted
    assert field_ids == ["custom_secret"]
    # The default admin-credential pattern is not applied when a custom
    # policy list is supplied.
    default_text = f"The synthetic test key is `{_ALPHA_ADMIN_KEY}`."
    redacted2, field_ids2 = redact_sensitive_fields(default_text, roles=[], policies=policies)
    assert redacted2 == default_text
    assert field_ids2 == []


# -- Auth-boundary milestone: redact_source_metadata --------------------------


def test_redact_source_metadata_redacts_a_matching_field_for_unauthorized_role():
    """attachment_name/section_path fields matching a sensitive pattern are redacted too."""
    fields = {"attachment_name": f"key-{_ALPHA_ADMIN_KEY}.pdf", "section_path": "Admin > Keys"}
    redacted, field_ids = redact_source_metadata(fields, roles=["tenant_alpha_operator"])
    assert _ALPHA_ADMIN_KEY not in redacted["attachment_name"]
    assert "[REDACTED:SENSITIVE_FIELD]" in redacted["attachment_name"]
    assert redacted["section_path"] == "Admin > Keys"
    assert field_ids == ["synthetic_admin_credential"]


def test_redact_source_metadata_preserves_fields_for_authorized_role():
    """An authorized role sees metadata fields unredacted."""
    fields = {"attachment_name": f"key-{_ALPHA_ADMIN_KEY}.pdf", "section_path": None}
    redacted, field_ids = redact_source_metadata(fields, roles=["tenant_alpha_admin"])
    assert redacted["attachment_name"] == fields["attachment_name"]
    assert field_ids == []


def test_redact_source_metadata_passes_through_none_values_unchanged():
    """A None-valued metadata field stays None, never coerced to a string."""
    fields = {"attachment_name": None, "section_path": None}
    redacted, field_ids = redact_source_metadata(fields, roles=[])
    assert redacted == {"attachment_name": None, "section_path": None}
    assert field_ids == []


# -- Auth-boundary milestone: find_duplicate_sensitive_occurrences ------------


def test_duplicate_secret_in_neighboring_chunk_is_also_tagged():
    """The same literal secret appearing in two separately-ingested chunks is flagged as duplicated.

    Requirement 7's concrete "secret appears in a neighboring/duplicate
    chunk" scenario -- built as an in-test fixture (per the approved
    design adjustment), not a file added to the canonical knowledge base.
    """
    chunks = [
        _chunk(
            "c1", f"The synthetic test key is {_ALPHA_ADMIN_KEY}.", ["synthetic_admin_credential"]
        ),
        _chunk(
            "c2",
            f"Reference copy of the same key: {_ALPHA_ADMIN_KEY}.",
            ["synthetic_admin_credential"],
        ),
    ]

    findings = find_duplicate_sensitive_occurrences(chunks)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.field_id == "synthetic_admin_credential"
    assert sorted(finding.chunk_ids) == ["c1", "c2"]
    assert finding.untagged_chunk_ids == []
    # The report never contains the raw literal -- only a hash.
    assert _ALPHA_ADMIN_KEY not in finding.literal_value_hash


def test_untagged_duplicate_is_flagged_by_detector():
    """A chunk containing the pattern but missing its ingestion-time tag is flagged."""
    chunks = [
        _chunk(
            "c1", f"The synthetic test key is {_ALPHA_ADMIN_KEY}.", ["synthetic_admin_credential"]
        ),
        _chunk("c2", f"Untagged copy: {_ALPHA_ADMIN_KEY}.", sensitive_field_ids=None),
    ]

    findings = find_duplicate_sensitive_occurrences(chunks)

    assert len(findings) == 1
    assert findings[0].untagged_chunk_ids == ["c2"]


def test_fully_tagged_duplicates_produce_no_findings():
    """Two occurrences of DIFFERENT secrets, each correctly tagged once, produce no findings."""
    chunks = [
        _chunk("c1", f"Alpha key: {_ALPHA_ADMIN_KEY}.", ["synthetic_admin_credential"]),
        _chunk("c2", "Nothing sensitive here at all."),
    ]

    findings = find_duplicate_sensitive_occurrences(chunks)

    assert findings == []


def test_single_correctly_tagged_occurrence_produces_no_finding():
    """A single, correctly-tagged occurrence (no duplicate, no missing tag) is not flagged."""
    chunks = [_chunk("c1", f"Key: {_ALPHA_ADMIN_KEY}.", ["synthetic_admin_credential"])]

    findings = find_duplicate_sensitive_occurrences(chunks)

    assert findings == []
