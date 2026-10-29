"""Typed data model for the Context Engineering subsystem.

`ContextItem` is the normalized unit every evidence source (vector
retrieval, GraphRAG, MCP/business tools, agent tool-fetched evidence,
conversation history) is converted into before budgeting, deduplication,
prioritization, compression, and packing ever run. Nothing downstream of
`rag.context.normalize` ever looks at a raw `SearchResult`/`CaseApproval`/
history turn again; it only ever sees `ContextItem`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Where a `ContextItem`'s content originated. Mirrors (but is not
#: identical to) `SearchResult.origin`: `"conversation_history"` and
#: `"system"`/`"query"` have no `SearchResult` equivalent at all, and a
#: graph path is its own unit rather than one `SearchResult` per hop.
ContextSourceType = Literal[
    "document",
    "graph_path",
    "mcp_tool",
    "tool_fetched",
    "conversation_history",
    "system",
    "query",
]


class GraphHop(BaseModel):
    """One hop of a graph path, kept structured so packing can render it as one unit.

    Attributes
    ----------
    subject, relation, object : str
        E.g. ``("Alpha API", "DEPENDS_ON", "Alpha Database")``.
    """

    subject: str
    relation: str
    object: str


class ContextItem(BaseModel):
    """One normalized, budget-aware unit of evidence eligible for the LLM prompt.

    Never constructed from un-authorized or un-redacted data: every
    `normalize_*` function in `rag.context.normalize` is called only on
    results that have already passed through
    `RetrievalPipeline.sanitize_evidence` (field redaction + injection
    flagging), the same invariant every pre-existing evidence path
    already upholds. This type adds no new authorization logic of its
    own; it is a packaging layer over already-authorized evidence.

    Attributes
    ----------
    content : str
        The text that would be rendered into the prompt. Mutated in
        place only by `rag.context.compression` (never by packing or
        prioritization), and compression never reintroduces text an
        upstream redaction pass removed (see that module's docstring).
    source_type : ContextSourceType
        Bounded vocabulary; never a raw provider/tool string.
    source_id : str
        Stable identifier within `source_type` (`chunk_id`, a graph
        path's synthetic id, an MCP case id, a conversation turn id).
    citation : str
        Human-readable provenance label, reusing
        `rag.retrieval.pipeline.source_label` for document/graph/
        tool-fetched items so citation wording never drifts between the
        legacy `build_context` path and this one.
    relevance_score : float | None
        The originating retrieval/search score (cosine similarity, BM25
        rank-fusion score, or a fixed constant for non-ranked evidence
        such as graph facts or tool-fetched chunks). `None` only for
        conversation history and the system/query items, which have no
        retrieval score at all.
    reranker_score : float | None
        Always `None` today: the existing `Reranker.rerank()` overwrites
        `SearchResult.score` in place rather than keeping the pre-rerank
        and post-rerank scores separately, so there is no distinct
        rerank score to carry here without changing `Reranker`'s public
        contract. Documented as a known limitation, not fabricated.
    authority_score : float | None
        Derived from `trust_level` (see `rag.context.prioritize`);
        `None` when no trust metadata exists for this item (the common
        case for a pre-governance-metadata document).
    freshness_status : str | None
        `document_version`/`status` governance field, when present.
    is_current : bool | None
        Whether this item's document is the currently-effective member
        of its version family, when freshness metadata exists.
    token_count : int
        Computed once by `rag.context.tokens.TokenCounter`, never
        estimated twice with different strategies within one request.
    tenant_id : str | None
        Carried through for observability/debugging only; this type
        performs no authorization of its own.
    dataset_id : str | None
    sensitive_field_ids : list[str]
        Field ids already redacted upstream (see
        `SearchResult.redacted_field_ids`); carried through so a
        caching layer can tell "this item had sensitive content redacted"
        without re-scanning text.
    injection_suspected : bool
        Carried through from `SearchResult.injection_suspected`/
        `detect_injection`; never used to drop an item (injection
        detection is reinforcement/telemetry only, per the existing
        `rag.retrieval.injection_detection` design), only to annotate it.
    document_id : str | None
    chunk_id : str | None
    content_type : str | None
        `ChunkMetadata.content_type` (`"prose"`/`"table"`/`"code"`/
        `"configuration"`/`"image"`/`"chart"`/`"graph_fact"`), when
        known. Drives `rag.context.compression`'s choice between
        sentence-level trimming (prose) and line-level trimming
        (table/code/configuration, where a sentence-boundary regex would
        mangle structure).
    graph_path : list[GraphHop]
        Populated only for `source_type == "graph_path"`; kept as
        structured hops so `rag.context.graph_packing` can render the
        whole path as one atomic unit rather than one `ContextItem` per
        hop (which would let packing split a path mid-traversal).
    tool_name : str | None
        Populated only for `source_type in {"mcp_tool", "tool_fetched"}`.
    origin : str | None
        Mirrors `SearchResult.origin`/`Citation.origin` for document/
        graph/tool items, so a `ContextItem` can still be traced back to
        its originating evidence path.
    original_token_count : int | None
        Token count before compression ran; `None` if this item was
        never compressed.
    compressed : bool
        Whether `rag.context.compression` rewrote `content` for this item.
    compression_strategy : str | None
        Which compression strategy ran (`"sentence_trim"`,
        `"section_select"`, `"extractive"`), `None` if uncompressed.
    priority_rank : int | None
        Set by `rag.context.prioritize`; `0` is highest priority. `None`
        before prioritization runs.
    merged_from : list[str]
        `source_id`s of other candidate items collapsed into this one by
        `rag.context.dedup` (provenance is kept, not discarded, when
        duplicates are collapsed). Empty for an item that was not a
        deduplication target.
    included : bool
        Set by `rag.context.packing`; whether this item made it into the
        final packed context. Every candidate item is kept in the trace
        (see `PackingTrace`) regardless of this flag, so a caller can
        inspect what was dropped and why.
    """

    content: str
    source_type: ContextSourceType
    source_id: str
    citation: str = ""
    relevance_score: float | None = None
    reranker_score: float | None = None
    authority_score: float | None = None
    freshness_status: str | None = None
    is_current: bool | None = None
    token_count: int = 0
    tenant_id: str | None = None
    dataset_id: str | None = None
    sensitive_field_ids: list[str] = Field(default_factory=list)
    injection_suspected: bool = False
    document_id: str | None = None
    chunk_id: str | None = None
    content_type: str | None = None
    graph_path: list[GraphHop] = Field(default_factory=list)
    tool_name: str | None = None
    origin: str | None = None
    original_token_count: int | None = None
    compressed: bool = False
    compression_strategy: str | None = None
    priority_rank: int | None = None
    merged_from: list[str] = Field(default_factory=list)
    included: bool = False


class ConversationTurn(BaseModel):
    """One caller-supplied conversation turn.

    Caller-asserted, not server-persisted: this system has no session
    store (see `rag.context.history`'s module docstring for why), so a
    multi-turn caller re-sends prior turns on each request, the same
    "caller asserts, server enforces/bounds" pattern this codebase
    already uses for `tenant_id`/`roles`/`case_approvals`.

    Attributes
    ----------
    role : {"user", "assistant"}
        Never `"system"` or `"tool"`: system instructions are never
        caller-suppliable (see `rag.context.history` module docstring),
        and tool outcomes are summarized into the assistant turn's own
        text by the caller, not modeled as a third role here.
    content : str
        Turn text. Never chain-of-thought; this is the caller-visible
        answer text a prior turn actually returned, or the user's prior
        message.
    turn_id : str | None
        Optional caller-supplied id (e.g. a prior `request_id`), for
        citation/debugging only.
    """

    role: Literal["user", "assistant"]
    content: str
    turn_id: str | None = None


class SourceBudget(BaseModel):
    """Token budget allocated to one evidence-source bucket.

    Attributes
    ----------
    source_type : str
        One of `"document"`, `"graph_path"`, `"mcp_tool"` (document +
        graph_path + mcp_tool together cover every non-history,
        non-tool_fetched evidence bucket; `tool_fetched` evidence shares
        the `document` bucket since it is drawn from the same corpus),
        or `"conversation_history"`.
    allocated_tokens : int
        The budget this bucket may spend.
    used_tokens : int
        Tokens actually spent by included items from this bucket, set
        after packing.
    """

    source_type: str
    allocated_tokens: int
    used_tokens: int = 0


class ContextBudget(BaseModel):
    """The resolved token budget for one request, before per-source allocation.

    See `rag.context.planner.ContextBudgetPlanner` for how this is computed.

    Attributes
    ----------
    model_context_window : int
        The model's total context window, from
        `rag.context.model_limits`.
    reserved_output_tokens : int
        Tokens reserved so generation itself is never starved.
    system_prompt_tokens : int
        Measured tokens in the rendered system prompt.
    user_query_tokens : int
        Measured tokens in the user's query text alone (not the full
        user turn, which also carries packed context).
    framework_overhead_tokens : int
        A configured, documented allowance for chat-template/tool-schema
        overhead the raw text token count doesn't capture (Ollama's
        `chat()` endpoint adds role/template tokens per message that a
        plain string token count underestimates).
    safety_margin_tokens : int
        An explicit buffer against estimation error (see
        `rag.context.tokens`'s documented estimation strategy).
    available_context_tokens : int
        `model_context_window - reserved_output_tokens -
        system_prompt_tokens - user_query_tokens -
        framework_overhead_tokens - safety_margin_tokens`, floored at 0.
    budget_deficit_tokens : int
        How far the fixed costs above exceed the window, or 0. Positive
        means the prompt cannot fit even with no evidence at all.
    source_budgets : dict[str, SourceBudget]
        Per-source allocation of `available_context_tokens`.
    """

    model_context_window: int
    reserved_output_tokens: int
    system_prompt_tokens: int
    user_query_tokens: int
    framework_overhead_tokens: int
    safety_margin_tokens: int
    available_context_tokens: int
    budget_deficit_tokens: int = 0
    source_budgets: dict[str, SourceBudget] = Field(default_factory=dict)

    @property
    def infeasible(self) -> bool:
        """Whether the mandatory system/query/output reservation alone overflows the window."""
        return self.budget_deficit_tokens > 0


class DeduplicationStats(BaseModel):
    """Counts from one deduplication pass. See `rag.context.dedup`."""

    candidates_before: int
    items_after: int
    tokens_saved: int
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0


class CompressionStats(BaseModel):
    """Counts from one compression pass. See `rag.context.compression`."""

    items_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def compression_ratio(self) -> float:
        """`1 - tokens_after / tokens_before`; `0.0` when nothing was compressed."""
        if self.tokens_before <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.tokens_after / self.tokens_before))


class PackingDecision(BaseModel):
    """One item's inclusion/exclusion decision, for the packing trace.

    Attributes
    ----------
    source_id : str
        The `ContextItem.source_id` this decision is about.
    source_type : str
    token_count : int
    included : bool
    reason : str
        A short, fixed-vocabulary reason (`"included"`,
        `"source_budget_exhausted"`, `"global_budget_exhausted"`,
        `"oversized_after_compression"`, `"duplicate"`).
    """

    source_id: str
    source_type: str
    token_count: int
    included: bool
    reason: str


class PackingTrace(BaseModel):
    """Full record of one packing run, for debugging/observability/tests."""

    strategy: Literal["top_n", "naive_token_first", "priority_aware"]
    decisions: list[PackingDecision] = Field(default_factory=list)
    truncation_count: int = 0
    dropped_item_count: int = 0

    @property
    def dropped_source_ids(self) -> list[str]:
        """`source_id`s of every excluded item, in decision order."""
        return [d.source_id for d in self.decisions if not d.included]


class PackedContext(BaseModel):
    """The final product of the Context Engineering pipeline.

    `rendered_context` is a drop-in replacement for
    `rag.retrieval.pipeline.build_context()`'s return value: callers
    substitute one string for the other with no other code change
    required at the prompt-rendering call site.

    Attributes
    ----------
    rendered_context : str
        The packed, `[Source N: ...]`-labeled text ready for
        `PromptTemplate.render(context=...)`.
    included_items : list[ContextItem]
        Items that made it into `rendered_context`, in render order.
    budget : ContextBudget
        The budget this was packed against.
    dedup_stats : DeduplicationStats
    compression_stats : CompressionStats
    packing_trace : PackingTrace
    context_utilization : float
        `packed_context_tokens / available_context_tokens`.
    context_reduction : float
        `1 - packed_tokens / candidate_tokens`.
    candidate_token_count : int
        Total tokens across every candidate item, before dedup/packing.
    packed_token_count : int
        Total tokens actually included in `rendered_context`.
    build_latency_ms : float
    truncated : bool
        Whether any eligible-but-unaffordable evidence had to be
        dropped to respect the budget (see `rag.context.engine`'s
        graceful-degradation handling).
    """

    rendered_context: str
    included_items: list[ContextItem] = Field(default_factory=list)
    budget: ContextBudget
    dedup_stats: DeduplicationStats
    compression_stats: CompressionStats
    packing_trace: PackingTrace
    context_utilization: float
    context_reduction: float
    candidate_token_count: int
    packed_token_count: int
    build_latency_ms: float = 0.0
    truncated: bool = False
