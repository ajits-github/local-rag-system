"""Field-level sensitive-value detection and redaction.

Distinct from, and layered on top of, document-level authorization
(`retrieval/authorization.py`): a chunk can be fully tenant/role-authorized
at the document level and still contain one specific field (e.g. an
administrator-only credential) that most authorized readers of that same
document must not see (see
`data/knowledge_base/security_evaluation/tenant_alpha/confidential-integration-runbook.md`,
whose ACL admits `tenant_alpha_operator`/`tenant_alpha_admin`/
`techfusion_support`, but whose own text restricts the literal admin key
to `tenant_alpha_admin` only). Document ACL decides "can this caller
retrieve the chunk at all"; this module decides "can this caller see this
particular value inside a chunk they were already allowed to retrieve."
`AuthorizationContext` itself is untouched -- this module only ever
consumes a plain `roles: list[str]`, kept as a conceptually separate
control per the field-level-safety milestone's design review.

Deliberately small and explicit ("prefer explicit metadata/annotations
over trying to infer all secrets dynamically"; "do not introduce a large
DLP product or external moderation framework"): a short, hardcoded,
documented list of `SensitiveFieldPolicy` entries, each a regex pattern
plus the caller roles allowed to see an unredacted match -- not a
general-purpose secret scanner, and not a config surface (these are
dataset-specific synthetic-secret shapes, not a per-deployment tunable;
same style as `injection_detection.py`'s `_INJECTION_PATTERNS`).
`detector: Literal["regex"]` is the swap point for a future non-regex
detector kind (e.g. a named-entity model) without changing this module's
public functions.

Deliberately **not** policing `TF-SYNTH-*` customer-correlation
identifiers: the runbook's own text says operators must correlate
failures using that identifier -- it is operational data support/
operators need, not a restricted field. Adding a policy for it would be
over-redaction (breaks the benign-regression check the milestone's design
review calls for).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class _ScannableChunk(Protocol):
    """Structural type for `find_duplicate_sensitive_occurrences` -- only what it reads.

    Deliberately a `Protocol`, not the concrete `Chunk` schema class:
    callers scanning chunks read from Postgres directly (see
    `scripts/detect_duplicate_sensitive_values.py`,
    `eval/run_eval.py`'s `_fetch_chunks_for_duplicate_scan`) build a
    lightweight duck-typed stand-in rather than a full `Chunk`, since
    there is no existing "fetch every chunk" `VectorStore` primitive to
    populate one from -- `Chunk` still satisfies this Protocol too.
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
    """One sensitivity annotation: a detectable pattern plus who may see it unredacted.

    Fields mirror the milestone's requested representation directly:
    `sensitivity_type` classifies the kind of value (e.g. "credential"),
    `allowed_roles` is the redaction policy (who is exempt), `field_id` is
    the optional-but-always-set identifier used for diagnosability in eval
    evidence (`eval/run_eval.py`'s `field_level_evidence`) and in
    `SearchResult.redacted_field_ids`. There is no `source_document`/
    `chunk` field on the policy itself -- scoping is implicit: a policy
    only ever runs against text from a chunk the caller was already
    document-level authorized to retrieve, so "which document" is never a
    property of the policy, only of the query that produced the chunk.
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
        # Two distinct literal shapes exist in the current corpus --
        # tenant_alpha's runbook uses SYNTHETIC_ONLY_ALPHA_KEY_7Q4M_DO_NOT_USE,
        # tenant_beta's uses SYNTHETIC_BETA_TOKEN_M8R5_NOT_VALID (confirmed by
        # grepping data/knowledge_base/security_evaluation for every
        # SYNTHETIC_* literal, not assumed) -- both alternatives live under
        # one policy since they're the same sensitivity_type/allowed_roles
        # concept (each tenant's own admin-only credential), and tenant
        # scoping already happened via document-level ACL before this
        # pattern ever runs.
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
    """Which policies' patterns match `text` -- ingestion-time tagging only, no role check.

    Called once per `ChunkSpan` by `ingestion/writer.py` and persisted as
    `ChunkMetadata.sensitive_field_ids`, so `redact_sensitive_fields`
    (query-time, role-aware) can skip untagged chunks without re-running
    every regex on every query.

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
    """Replace matched spans the caller isn't authorized for with a stable marker.

    Fail-closed by construction: `roles=[]` (no asserted identity) matches
    no policy's `allowed_roles`, so every policy that matches `text` gets
    redacted -- there is deliberately no "no roles means unrestricted"
    branch here, unlike document-level `AuthorizationContext` semantics.
    Callers decide *whether* to invoke this at all (see
    `retrieval/pipeline.py`'s `config.security.field_redaction.enabled`
    gate); once invoked, it never treats a missing identity as permission.

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


def redact_source_metadata(
    fields: dict[str, str | None],
    roles: list[str],
    policies: list[SensitiveFieldPolicy] | None = None,
) -> tuple[dict[str, str | None], list[str]]:
    """Apply the same sensitive-field redaction to metadata fields, not just chunk content.

    Auth-boundary milestone: a field-level-redacted value can still leak
    indirectly through a *permitted* document's own metadata (e.g. an
    `attachment_name` or `section_path` that echoes the value), even
    though `content` itself was correctly redacted. Reuses
    `redact_sensitive_fields` per field so the same fail-closed role check
    and pattern set apply uniformly.

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

    `literal_value_hash` is a sha256 digest of the matched substring --
    never the raw literal itself, so this diagnostic's own output (a
    report, a metric, a test assertion) can never leak a secret.
    """

    literal_value_hash: str
    field_id: str
    chunk_ids: list[str]
    untagged_chunk_ids: list[str]


def find_duplicate_sensitive_occurrences(
    chunks: Sequence[_ScannableChunk], policies: list[SensitiveFieldPolicy] | None = None
) -> list[DuplicateSensitiveOccurrence]:
    """Find sensitive literals present in multiple chunks, or missing their ingestion-time tag.

    Diagnostic/validation only -- run against a corpus's chunks (e.g. by
    `scripts/detect_duplicate_sensitive_values.py`), not part of the
    query-time enforcement path. Catches two related gaps: (1) the same
    protected literal value copy-pasted into a second document/chunk that
    never got scanned together with the first (a true duplicate), and (2)
    a chunk whose content matches a policy pattern but whose
    `ChunkMetadata.sensitive_field_ids` doesn't include that `field_id` --
    an ingestion-time tagging miss, which would let query-time redaction
    (`redact_sensitive_fields`, gated on the tag as a cheap pre-check --
    see `retrieval/pipeline.py`) skip that chunk's redaction pass
    entirely.

    Parameters
    ----------
    chunks : Sequence[_ScannableChunk]
        Chunks to scan (their own `.content` and `.metadata.sensitive_field_ids`)
        -- any object structurally matching this shape, including a real
        `Chunk` or a lightweight duck-typed stand-in.
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
