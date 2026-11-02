"""Provider-egress policy: gates retrieved content before it reaches an external provider.

Two confirmed hosted-egress points in this codebase use this gate:
`run_ragas_eval.py`'s context-building step (sends retrieved chunk content
to the judge LLM, `config.judge.provider` in `openai`/`anthropic`), and
`RetrievalPipeline`'s hosted reranker call (`config.reranker.provider ==
"cohere"`, see `retrieval/pipeline.py`'s `_retrieve_timed_inner`).
Production `answer()`'s generation step never calls a hosted LLM
(`config.generation.provider` is `ollama`-only), so no other call site
needs this gate.

A third checkpoint layered on top of document-level authorization and
field-level redaction (not a replacement for either), applied to the same
per-source dict `RetrievalPipeline.answer()`/`source_dict()` returns.

Lives under `rag.retrieval`, not `rag.eval`, so `RetrievalPipeline` can
use it directly without violating this codebase's
`ingestion`/`retrieval`/`generation`-never-imports-`rag.eval` rule;
`rag.eval.egress_policy` re-exports this module unchanged for every
existing eval caller.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from rag.config import AppConfig, EgressPolicyConfig


class EgressDecision(BaseModel):
    """The outcome of checking one retrieved source against the egress policy."""

    allowed: bool
    redacted_context: str
    blocked_reason: str | None = None


def _governance_block_reason(governance: dict[str, Any], policy: EgressPolicyConfig) -> str | None:
    """Return why one source's tenant, classification or trust forbids egress, if any."""
    tenant_id = governance.get("tenant_id")
    if tenant_id is not None and tenant_id in policy.blocked_tenant_ids:
        return "blocked_tenant"

    classification = governance.get("classification")
    if classification is None:
        if not policy.allow_unclassified:
            return "unclassified"
    elif not policy.classification_policy.get(classification, False):
        return "classification"

    if policy.require_authoritative_trust and governance.get("trust_level") != "authoritative":
        return "trust_level"
    return None


def apply_egress_policy(source: dict[str, Any], config: AppConfig) -> EgressDecision:
    """Decide whether `source`'s content may leave the local environment.

    A no-op (`allowed=True`, content unchanged) when
    `config.security.egress_policy.enabled` is `False`. When enabled,
    checked in order:

    1. `source["tenant_id"]` against `blocked_tenant_ids`. An explicit
       tenant deny-list.
    2. `source["classification"]` against `classification_policy`, which
       **fails closed**: a classification with no entry in that dict is
       blocked, not silently allowed (`.get(classification, False)`).
       A missing classification is allowed only when `allow_unclassified`
       is `True`; otherwise it is blocked as `"unclassified"`.
    3. `require_authoritative_trust`. When `True`, only
       `trust_level == "authoritative"` may egress.
    4. `block_unredacted_sensitive_fields`. When `True`, any source
       whose `sensitive_field_ids` isn't a subset of its
       `redacted_field_ids` (i.e. a tagged sensitive field that wasn't
       actually redacted for this particular retrieval) is blocked
       outright, regardless of why it wasn't redacted.

    Checks 1 to 3 also run on each `source["support_governance"]` entry, so a
    multi-source graph fact egresses only if every document it depends on may.

    Parameters
    ----------
    source : dict[str, Any]
        One entry from `RetrievalPipeline.answer()`'s `"sources"` list
        (or the dense/bm25/fused lists from `retrieve_attribution`, or a
        pre-rerank candidate rendered via `source_dict()`); reads
        `content`/`tenant_id`/`classification`/`trust_level`/
        `sensitive_field_ids`/`redacted_field_ids`/`support_governance`.
    config : AppConfig
        Application configuration.

    Returns
    -------
    EgressDecision
        `allowed=False` sources carry an empty `redacted_context`. The
        caller must exclude them from anything sent to a hosted provider,
        never send a placeholder derived from the blocked content.
    """
    policy = config.security.egress_policy
    content = source.get("content") or ""
    if not policy.enabled:
        return EgressDecision(allowed=True, redacted_context=content)

    for governance in (source, *(source.get("support_governance") or ())):
        reason = _governance_block_reason(governance, policy)
        if reason is not None:
            return EgressDecision(allowed=False, redacted_context="", blocked_reason=reason)

    if policy.block_unredacted_sensitive_fields:
        sensitive_ids = set(source.get("sensitive_field_ids") or [])
        redacted_ids = set(source.get("redacted_field_ids") or [])
        if sensitive_ids - redacted_ids:
            return EgressDecision(
                allowed=False, redacted_context="", blocked_reason="unredacted_sensitive_field"
            )

    return EgressDecision(allowed=True, redacted_context=content)
