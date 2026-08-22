"""Field-level sensitive-value detection and redaction.

Document-level authorization decides whether a caller may retrieve a
chunk at all. This module decides whether a specific value inside an
already-authorized chunk (e.g. an admin-only credential) must be redacted
for that caller's roles.

Policies are a short, hardcoded list of pattern-plus-allowed-roles
entries, not a general-purpose secret scanner or config surface.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class _ScannableChunk(Protocol):
    """Minimal chunk shape read by `find_duplicate_sensitive_occurrences`.

    A `Protocol` rather than the concrete `Chunk` class so callers can
    pass a lightweight duck-typed stand-in when scanning chunks read
    directly from the database.
    """

    @property
    def id(self) -> str: ...  # noqa: D102

    @property
    def content(self) -> str: ...  # noqa: D102

    @property
    def metadata(self) -> _ScannableChunkMetadata: ...  # noqa: D102


class _ScannableChunkMetadata(Protocol):
    """Structural type for the `.metadata.sensitive_field_ids` this module reads."""

    @property
    def sensitive_field_ids(self) -> list[str] | None: ...  # noqa: D102


class SensitiveFieldPolicy(BaseModel):
    """A detectable sensitive-value pattern plus who may see it unredacted.

    Parameters
    ----------
    field_id : str
        Identifier used in diagnostics and `SearchResult.redacted_field_ids`.
    sensitivity_type : str
        Category of the value, e.g. "credential".
    detector : {"regex"}
        Detection method. Currently only regex matching is supported.
    pattern : str
        Regex pattern identifying the sensitive value.
    allowed_roles : list[str]
        Roles exempt from redaction for this field.
    redaction_marker : str
        Text substituted for an unauthorized match.

    Notes
    -----
    A policy runs only against text from a chunk the caller was already
    document-level authorized to retrieve.
    """

    field_id: str
    sensitivity_type: str
    detector: Literal["regex"] = "regex"
    pattern: str
    allowed_roles: list[str] = Field(default_factory=list)
    redaction_marker: str = "[REDACTED:SENSITIVE_FIELD]"


DEFAULT_FIELD_POLICIES: list[SensitiveFieldPolicy] = [
    SensitiveFieldPolicy(
        field_id="synthetic_admin_credential",
        sensitivity_type="credential",
        # Alpha and Beta use different literal formats for the same
        # tenant-admin-credential policy; tenant scoping already happened
        # via document-level ACL before this pattern ever runs.
        pattern=r"SYNTHETIC_ONLY_\w+|SYNTHETIC_BETA_TOKEN_\w+",
        allowed_roles=[
            "tenant_alpha_admin",
            "tenant_beta_admin",
            "security_admin",
            "system_admin",
        ],
    ),
    SensitiveFieldPolicy(
        field_id="synthetic_internal_token",
        sensitivity_type="credential",
        pattern=r"SYNTHETIC_INTERNAL_TOKEN_\w+",
        allowed_roles=["security_admin", "system_admin"],
    ),
]


def detect_sensitive_field_ids(
    text: str, policies: list[SensitiveFieldPolicy] | None = None
) -> list[str]:
    """Return the policies whose pattern matches `text`.

    Ingestion-time tagging only; does not check roles. Persisted as
    `ChunkMetadata.sensitive_field_ids` so query-time redaction can skip
    untagged chunks without re-running every pattern.

    Parameters
    ----------
    text : str
        A chunk span's text.
    policies : list[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    list[str]
        `field_id`s of every policy whose pattern matches `text` at least once.
    """
    active = policies if policies is not None else DEFAULT_FIELD_POLICIES
    return [p.field_id for p in active if re.search(p.pattern, text)]


def is_role_authorized_for_field(
    field_id: str, roles: list[str], policies: list[SensitiveFieldPolicy] | None = None
) -> bool:
    """Whether any of `roles` is allowed to see `field_id` unredacted.

    Parameters
    ----------
    field_id : str
        A `SensitiveFieldPolicy.field_id`.
    roles : list[str]
        Caller roles to check.
    policies : list[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    bool
        True if `field_id` isn't a known policy (nothing to restrict) or
        `roles` intersects that policy's `allowed_roles`.
    """
    active = policies if policies is not None else DEFAULT_FIELD_POLICIES
    policy = next((p for p in active if p.field_id == field_id), None)
    if policy is None:
        return True
    return any(role in policy.allowed_roles for role in roles)


def redact_sensitive_fields(
    text: str, roles: list[str], policies: list[SensitiveFieldPolicy] | None = None
) -> tuple[str, list[str]]:
    """Replace matched spans the caller isn't authorized for with a redaction marker.

    Fail-closed: an empty `roles` list matches no policy's
    `allowed_roles`, so every matching policy gets redacted rather than
    treated as unrestricted.

    Parameters
    ----------
    text : str
        Chunk content to scan.
    roles : list[str]
        Caller roles asserted for this query (empty if no identity known).
    policies : list[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    tuple[str, list[str]]
        The (possibly modified) text, and the `field_id`s of every policy
        that redacted at least one match.
    """
    active = policies if policies is not None else DEFAULT_FIELD_POLICIES
    redacted_ids: list[str] = []
    for policy in active:
        if any(role in policy.allowed_roles for role in roles):
            continue
        new_text, n = re.subn(policy.pattern, policy.redaction_marker, text)
        if n:
            text = new_text
            redacted_ids.append(policy.field_id)
    return text, redacted_ids


_REDACTION_MARKER_ECHO_REPLACEMENT = "this value is unavailable at your access level"


def sanitize_redaction_markers_in_answer(
    text: str, policies: Sequence[SensitiveFieldPolicy] | None = None
) -> str:
    """Replace any literal redaction-marker text a generation model echoed back.

    `redact_sensitive_fields` removes the raw sensitive value before it
    reaches the model, but the model can still copy the internal marker
    into its final answer. This deterministic backstop keeps caller-facing
    prose free of marker syntax, independent of prompt compliance.

    Parameters
    ----------
    text : str
        A generated answer, from either the classic or agentic path.
    policies : Sequence[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    str
        `text` with every literal occurrence of a configured
        `redaction_marker` (optionally wrapped in backticks or quotes the
        model added around it) replaced by a caller-facing paraphrase.
    """
    active = policies if policies is not None else DEFAULT_FIELD_POLICIES
    markers = {p.redaction_marker for p in active}
    for marker in markers:
        wrapped_pattern = r"[`'\"]*" + re.escape(marker) + r"[`'\"]*"
        text = re.sub(wrapped_pattern, _REDACTION_MARKER_ECHO_REPLACEMENT, text)
    return text


def redact_source_metadata(
    fields: dict[str, str | None],
    roles: list[str],
    policies: list[SensitiveFieldPolicy] | None = None,
) -> tuple[dict[str, str | None], list[str]]:
    """Apply the same sensitive-field redaction to metadata field values.

    Guards against a value leaking indirectly through metadata (e.g. an
    `attachment_name` that echoes the value) even when chunk content was
    already redacted.

    Parameters
    ----------
    fields : dict[str, str | None]
        Metadata field name -> value (e.g. `{"attachment_name": ...,
        "section_path": ...}`). `None` values pass through unchanged.
    roles : list[str]
        Caller roles asserted for this query (empty if no identity known).
    policies : list[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    tuple[dict[str, str | None], list[str]]
        The (possibly modified) fields dict, and the deduplicated
        `field_id`s of every policy that redacted at least one field.
    """
    redacted_fields: dict[str, str | None] = {}
    all_redacted_ids: list[str] = []
    for name, value in fields.items():
        if value is None:
            redacted_fields[name] = None
            continue
        redacted_text, redacted_ids = redact_sensitive_fields(value, roles, policies)
        redacted_fields[name] = redacted_text
        for field_id in redacted_ids:
            if field_id not in all_redacted_ids:
                all_redacted_ids.append(field_id)
    return redacted_fields, all_redacted_ids


class DuplicateSensitiveOccurrence(BaseModel):
    """One sensitive literal value found in more than one chunk, or tagged inconsistently.

    Parameters
    ----------
    literal_value_hash : str
        Sha256 digest of the matched substring, not the raw literal, so
        this diagnostic's own output can never leak a secret.
    field_id : str
        The `SensitiveFieldPolicy.field_id` that matched.
    chunk_ids : list[str]
        Chunks containing this literal value.
    untagged_chunk_ids : list[str]
        Subset of `chunk_ids` missing the corresponding ingestion-time tag.
    """

    literal_value_hash: str
    field_id: str
    chunk_ids: list[str]
    untagged_chunk_ids: list[str]


def find_duplicate_sensitive_occurrences(
    chunks: Sequence[_ScannableChunk], policies: list[SensitiveFieldPolicy] | None = None
) -> list[DuplicateSensitiveOccurrence]:
    """Find sensitive literals present in multiple chunks, or missing their ingestion-time tag.

    Diagnostic only, not part of query-time enforcement. Catches two
    gaps: the same sensitive value copied into a second chunk that was
    never cross-checked against the first, and a chunk whose content
    matches a policy pattern but isn't tagged with that `field_id`.

    Parameters
    ----------
    chunks : Sequence[_ScannableChunk]
        Chunks to scan.
    policies : list[SensitiveFieldPolicy] | None, optional
        Defaults to `DEFAULT_FIELD_POLICIES`.

    Returns
    -------
    list[DuplicateSensitiveOccurrence]
        One entry per `(field_id, literal_value_hash)` pair that either
        appears in more than one chunk, or has at least one untagged
        occurrence. Empty when the corpus is fully and uniquely tagged.
    """
    active = policies if policies is not None else DEFAULT_FIELD_POLICIES
    chunk_ids_by_key: dict[tuple[str, str], list[str]] = {}
    untagged_ids_by_key: dict[tuple[str, str], list[str]] = {}

    for chunk in chunks:
        tagged_ids = set(chunk.metadata.sensitive_field_ids or [])
        for policy in active:
            for match in re.finditer(policy.pattern, chunk.content):
                value_hash = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
                key = (policy.field_id, value_hash)
                chunk_ids = chunk_ids_by_key.setdefault(key, [])
                if chunk.id not in chunk_ids:
                    chunk_ids.append(chunk.id)
                if policy.field_id not in tagged_ids:
                    untagged_ids = untagged_ids_by_key.setdefault(key, [])
                    if chunk.id not in untagged_ids:
                        untagged_ids.append(chunk.id)

    findings: list[DuplicateSensitiveOccurrence] = []
    for (field_id, value_hash), chunk_ids in chunk_ids_by_key.items():
        untagged_ids = untagged_ids_by_key.get((field_id, value_hash), [])
        if len(chunk_ids) > 1 or untagged_ids:
            findings.append(
                DuplicateSensitiveOccurrence(
                    literal_value_hash=value_hash,
                    field_id=field_id,
                    chunk_ids=sorted(chunk_ids),
                    untagged_chunk_ids=sorted(untagged_ids),
                )
            )
    return findings
