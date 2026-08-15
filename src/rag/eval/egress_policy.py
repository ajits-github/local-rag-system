"""Provider-egress policy: gates retrieved content before it reaches a hosted LLM.

Applies at the one confirmed hosted-egress point in this codebase:
`run_ragas_eval.py`'s context-building step, which hands retrieved chunk
content to `config.judge.provider` (`openai`/`anthropic`) for RAGAS
scoring. Production `answer()` never calls a hosted LLM --
`config.generation.provider` is `Literal["ollama"]` only -- so no other
call site needs this gate today; if a future hosted `generation`/`vision`
provider is added, it should call `apply_egress_policy` the same way.

Independent of, and layered on top of, whatever document-level
authorization and field-level redaction already ran to produce the
`source` dict this module inspects (the same per-source dict shape
`RetrievalPipeline.answer()` returns) -- this is a *third* checkpoint, not
a replacement for either.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from rag.config import AppConfig


class EgressDecision(BaseModel):
    """The outcome of checking one retrieved source against the egress policy."""

    allowed: bool
    redacted_context: str
    blocked_reason: str | None = None


def apply_egress_policy(source: dict[str, Any], config: AppConfig) -> EgressDecision:
    """Decide whether `source`'s content may leave the local environment.

    A no-op (`allowed=True`, content unchanged) when
    `config.security.egress_policy.enabled` is `False`. When enabled,
    checked in order:

    1. `source["tenant_id"]` against `blocked_tenant_ids` -- an explicit
       tenant deny-list.
    2. `source["classification"]` against `classification_policy` --
       **fails closed**: a classification with no entry in that dict is
       blocked, not silently allowed (`.get(classification, False)`).
       `None`/missing classification is treated as allowed, matching
       `"internal"`'s default, for the pre-governance-metadata corpus.
    3. `require_authoritative_trust` -- when `True`, only
       `trust_level == "authoritative"` may egress.
    4. `block_unredacted_sensitive_fields` -- when `True`, any source
       whose `sensitive_field_ids` isn't a subset of its
       `redacted_field_ids` (i.e. a tagged sensitive field that wasn't
       actually redacted for this particular retrieval) is blocked
       outright, regardless of why it wasn't redacted.

    Parameters
    ----------
    source : dict[str, Any]
        One entry from `RetrievalPipeline.answer()`'s `"sources"` list
        (or the dense/bm25/fused lists from `retrieve_attribution`) --
        reads `content`/`tenant_id`/`classification`/`trust_level`/
        `sensitive_field_ids`/`redacted_field_ids`.
    config : AppConfig
        Application configuration.

    Returns
    -------
    EgressDecision
        `allowed=False` sources carry an empty `redacted_context` -- the
        caller must exclude them from anything sent to a hosted provider,
        never send a placeholder derived from the blocked content.
    """
    policy = config.security.egress_policy
    content = source.get("content") or ""
    if not policy.enabled:
        return EgressDecision(allowed=True, redacted_context=content)

    tenant_id = source.get("tenant_id")
    if tenant_id is not None and tenant_id in policy.blocked_tenant_ids:
        return EgressDecision(allowed=False, redacted_context="", blocked_reason="blocked_tenant")

    classification = source.get("classification")
    if classification is not None and not policy.classification_policy.get(classification, False):
        return EgressDecision(allowed=False, redacted_context="", blocked_reason="classification")

    if policy.require_authoritative_trust and source.get("trust_level") != "authoritative":
        return EgressDecision(allowed=False, redacted_context="", blocked_reason="trust_level")

    if policy.block_unredacted_sensitive_fields:
        sensitive_ids = set(source.get("sensitive_field_ids") or [])
        redacted_ids = set(source.get("redacted_field_ids") or [])
        if sensitive_ids - redacted_ids:
            return EgressDecision(
                allowed=False, redacted_context="", blocked_reason="unredacted_sensitive_field"
            )

    return EgressDecision(allowed=True, redacted_context=content)
