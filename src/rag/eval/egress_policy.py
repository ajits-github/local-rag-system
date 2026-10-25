"""Backward-compatible re-export of `rag.retrieval.egress_policy`.

The egress-policy gate moved to `rag.retrieval` so `RetrievalPipeline`
could apply it directly to the hosted-reranker boundary
(`config.reranker.provider == "cohere"`), without violating this
codebase's `ingestion`/`retrieval`/`generation`-never-imports-`rag.eval`
rule. Every existing eval caller keeps importing from here unchanged.
"""

from __future__ import annotations

from rag.retrieval.egress_policy import EgressDecision, apply_egress_policy

__all__ = ["EgressDecision", "apply_egress_policy"]
