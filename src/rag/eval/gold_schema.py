"""Gold-data schema and path-suffix matching for retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from rag.path_matching import source_matches_relevant as _shared_source_matches_relevant


class GoldExample(BaseModel):
    """Represent one row of a gold JSONL evaluation file.

    Attributes
    ----------
    question : str
        The evaluation question.
    expected_answer : str or None
        Reference answer text, if authored.
    relevant_documents : list[str]
        Paths relative to the knowledge-base root (not document IDs,
        which are assigned at ingestion time). Matched against a
        retrieved chunk's stored `source` by path suffix.
    question_type : str or None
        Free-form question category.
    difficulty : str or None
        Free-form difficulty label.
    unanswerable : bool
        Whether the question has no valid answer in the corpus.
    content_type : str or None
        Authored ground-truth question category (e.g. "image_only",
        "relationship_aware"), distinct from `eval/content_type.py`'s
        chunker-derived document buckets.
    reference_contexts : list[str]
        Verbatim textual excerpts expected to support the answer.
    reference_visual_contexts : list[str]
        Ground truth for facts visually present in an image but absent
        from its caption or surrounding text.
    relevant_images : list[str]
        Image assets associated with the question.
    relevant_sections : list[str]
        Section paths associated with the question.
    requires_vision : bool
        Whether the question cannot be answered from text/caption alone.
    requires_relationship_expansion : bool
        Whether the question depends on relationship-expanded context.
    safety_category : str or None
        Category of safety scenario tested (e.g. cross-tenant access).
    user_tenant : str or None
        Simulated caller tenant, used to build an `AuthorizationContext`.
    user_roles : list[str]
        Simulated caller roles, used to build an `AuthorizationContext`.
    allowed_documents : list[str]
        Documents the simulated caller should be able to retrieve.
    forbidden_documents : list[str]
        Documents the simulated caller must not retrieve.
    expected_behavior : str or None
        Expected model behavior (e.g. refusal), for refusal metrics.
    requires_current_document : bool
        Whether the question requires the current document version.
    expected_document_version : str or None
        Expected resolved document version.
    expected_effective_date : str or None
        Expected effective date of the resolved version.
    query_as_of : str or None
        Simulated `as_of` date for freshness resolution.
    injection_present : bool
        Whether the scenario includes an embedded prompt-injection attempt.
    injection_source : str or None
        Document containing the injection attempt, if any.
    sensitive_data_present : bool
        Whether the scenario involves a field-level-redacted value.
    requires_authorization_filter : bool
        Whether the scenario depends on document-level authorization.
    requires_tenant_filter : bool
        Whether the scenario depends on tenant scoping.
    requires_trust_filter : bool
        Whether the scenario depends on trust-level filtering.
    expected_trust_level : str or None
        Trust level the query should require.
    requires_query_decomposition : bool
        Agentic RAG milestone: whether answering requires the agent to
        split the question into subquestions (`decompose` node).
    requires_multiple_retrieval_calls : bool
        Agentic RAG milestone: whether answering genuinely requires more
        than one tool dispatch, not just one broad retrieval.
    requires_latest_document_tool : bool
        Agentic RAG milestone: whether answering requires resolving a
        version family via the `get_latest_document` tool specifically
        (as opposed to freshness resolution inside plain `retrieve()`).
    expects_insufficient_evidence_retry : bool
        Agentic RAG milestone: whether a first retrieval is expected to
        miss, requiring the bounded reformulate-and-retry loop to find
        the answer.
    tool_not_needed : bool
        Agentic RAG milestone: whether this question should route to
        `classic_rag` even when `config.agent.enabled=True`. Used to
        score `unnecessary_agent_rate`.
    expects_max_step_termination : bool
        Agentic RAG milestone: whether this question is expected to
        exhaust `max_agent_steps`/`max_retrieval_attempts`/`max_tool_calls`
        rather than resolve normally.
    adversarial_tool_instruction : bool
        Agentic RAG milestone: whether a tool's output (not the original
        query) carries a prompt-injection payload. Tests that tool
        output stays evidence, never becomes an instruction.
    expected_tool_sequence : list[str]
        Agentic RAG milestone: the expected ordered tool names for
        `tool_selection_accuracy` scoring.
    agentic_category : str or None
        Agentic RAG milestone: authored scenario bucket (e.g.
        "query_decomposition", "latest_document_resolution",
        "retrieval_reformulation", "tool_not_needed",
        "insufficient_evidence", "adversarial_tool_output"). Distinct
        from `content_type`'s multimodal buckets and `safety_category`'s
        security buckets; used for `run_agent_eval.py`'s per-category
        breakdown.
    agentic_rationale : str or None
        Agentic RAG milestone: one-sentence authored explanation of why
        this question requires the tagged agentic behavior.
    expected_subquestions : list[str]
        Agentic RAG milestone: reference subquestions for scoring
        `decompose` node output quality.
    expected_reformulation : str or None
        Agentic RAG milestone: a reference reformulated query, for
        scenarios where the first phrasing is expected to miss and the
        `evaluate_evidence` node's retry should look more like this.
    minimum_expected_tool_calls, maximum_expected_tool_calls : int or None
        Agentic RAG milestone: the expected reasonable range of total
        tool dispatches for this question. `0`/`0` for `tool_not_needed`
        rows. Used to flag an agent run as under- or over-working a
        question, distinct from the hard `config.agent.max_tool_calls`
        ceiling.
    requires_layout_awareness : bool
        Layout-aware-ingestion milestone: whether answering depends on
        structure the classic text-only pipeline would have lost (a
        table/code/configuration block kept atomic, or a page/section
        boundary), distinct from `requires_vision` (image understanding)
        and `requires_relationship_expansion` (parent-prose linking).
    expected_content_types : list[str]
        Layout-aware-ingestion milestone: the `ChunkMetadata.content_type`
        value(s) (e.g. `"table"`/`"code"`/`"configuration"`/`"image"`)
        the supporting evidence is expected to carry. Distinct from
        `content_type` (this row's own authored question-category label).
    relevant_pages : list[int]
        Layout-aware-ingestion milestone: source page number(s) (PDF) or
        manual-page-break count (DOCX) the supporting evidence is
        expected to come from -- see `ChunkMetadata.page`.
    visual_question_type : str or None
        Layout-aware-ingestion milestone: authored sub-category for a
        vision-dependent question (e.g. `"chart_reading"`,
        `"diagram_relationship"`), when `requires_vision` is true.
    evidence_mode : str or None
        Layout-aware-ingestion milestone: how the question's supporting
        evidence is expected to be found -- e.g. `"text_only"`,
        `"vision_required"`, `"layout_structure"`.
    source_format : str or None
        Layout-aware-ingestion milestone: the source document's file
        format (e.g. `"pdf"`/`"docx"`/`"markdown"`), for per-format
        metric breakdowns independent of `ChunkMetadata.source_type`
        (which is derived from the retrieved chunk, not authored).

    Notes
    -----
    `reference_contexts`/`reference_visual_contexts` are evaluation-only
    ground truth: never indexed, never passed to ingestion, retrieval, or
    generation. All fields beyond `question`/`expected_answer`/
    `relevant_documents`/`question_type`/`difficulty`/`unanswerable` are
    optional so older gold files without them keep parsing unchanged.
    """

    question: str
    expected_answer: str | None = None
    relevant_documents: list[str] = Field(default_factory=list)
    question_type: str | None = None
    difficulty: str | None = None
    unanswerable: bool = False
    content_type: str | None = None
    reference_contexts: list[str] = Field(default_factory=list)
    reference_visual_contexts: list[str] = Field(default_factory=list)
    relevant_images: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    requires_vision: bool = False
    requires_relationship_expansion: bool = False
    safety_category: str | None = None
    user_tenant: str | None = None
    user_roles: list[str] = Field(default_factory=list)
    allowed_documents: list[str] = Field(default_factory=list)
    forbidden_documents: list[str] = Field(default_factory=list)
    expected_behavior: str | None = None
    requires_current_document: bool = False
    expected_document_version: str | None = None
    expected_effective_date: str | None = None
    query_as_of: str | None = None
    injection_present: bool = False
    injection_source: str | None = None
    sensitive_data_present: bool = False
    requires_authorization_filter: bool = False
    requires_tenant_filter: bool = False
    requires_trust_filter: bool = False
    expected_trust_level: str | None = None
    requires_query_decomposition: bool = False
    requires_multiple_retrieval_calls: bool = False
    requires_latest_document_tool: bool = False
    expects_insufficient_evidence_retry: bool = False
    tool_not_needed: bool = False
    expects_max_step_termination: bool = False
    adversarial_tool_instruction: bool = False
    expected_tool_sequence: list[str] = Field(default_factory=list)
    agentic_category: str | None = None
    agentic_rationale: str | None = None
    expected_subquestions: list[str] = Field(default_factory=list)
    expected_reformulation: str | None = None
    minimum_expected_tool_calls: int | None = None
    maximum_expected_tool_calls: int | None = None
    requires_layout_awareness: bool = False
    expected_content_types: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    visual_question_type: str | None = None
    evidence_mode: str | None = None
    source_format: str | None = None


def load_gold_jsonl(path: str | Path) -> list[GoldExample]:
    """Parse a gold JSONL file into `GoldExample` rows, skipping blank lines.

    Parameters
    ----------
    path : str | Path
        Path to the gold JSONL file.

    Returns
    -------
    list[GoldExample]
        One `GoldExample` per non-blank line.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldExample.model_validate(json.loads(line)))
    return examples


def source_matches_relevant(retrieved_source: str, relevant_path: str) -> bool:
    """Check whether `relevant_path`'s segments trail `retrieved_source`'s segments.

    For example, gold's "knowledge_base/security/x.md" matches a stored source of
    "data/knowledge_base/security/x.md" regardless of the "data/" root,
    without either side needing to hardcode that root. Re-exported from
    `rag.path_matching` (moved there so `rag.retrieval.freshness` can reuse
    the identical rule without importing from `rag.eval`.

    Parameters
    ----------
    retrieved_source : str
        A chunk's stored `source` path.
    relevant_path : str
        A gold example's relative `relevant_documents` entry.

    Returns
    -------
    bool
        True if `relevant_path` is a trailing subsequence of `retrieved_source`.
    """
    return _shared_source_matches_relevant(retrieved_source, relevant_path)


def normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace, for reference-context substring matching.

    Deliberately permissive on whitespace/case only, not semantics --
    `reference_contexts` entries are authored as verbatim excerpts from the
    source corpus, so this only smooths
    over incidental formatting differences (line wraps, trailing spaces),
    never paraphrase differences. See `reference_context_is_supported`'s
    documented limitation.

    Parameters
    ----------
    text : str
        Raw text to normalize.

    Returns
    -------
    str
        Lowercased, whitespace-collapsed text.
    """
    return " ".join(text.split()).lower()


def reference_context_is_supported(reference: str, candidate_texts: Iterable[str]) -> bool:
    """Check whether `reference` is a normalized substring of any `candidate_texts` entry.

    Used both by `scripts/validate_gold_file.py` (reference resolves to the
    source corpus) and `eval/run_eval.py` (reference resolves to a
    retrieved chunk): same matching rule, different candidate sets.

    **Limitation** (documented, not hidden): this is substring containment
    after whitespace normalization, not semantic similarity. It is valid
    because `reference_contexts` entries are authored as verbatim excerpts
    (confirmed against real gold rows: exact JSON blocks, exact backtick
    commands, exact caption sentences including their `*asterisks*`), but
    it is brittle to any *other* formatting drift between the gold
    author's copy and the actual text being checked against. For example,
    a reference spanning a table row-group boundary can get split across
    two persisted chunks. This under-counts (produces false negatives on
    genuinely-supported references), never over-counts a coincidental
    match into a false positive at this excerpt length in practice, but is
    not a proof of "the model actually used this passage."

    Parameters
    ----------
    reference : str
        One `reference_contexts` entry.
    candidate_texts : Iterable[str]
        Texts to search within (source document text, or retrieved chunk
        contents).

    Returns
    -------
    bool
        True if the normalized `reference` is a substring of any
        normalized `candidate_texts` entry. False (never a match) for a
        blank/whitespace-only `reference`.
    """
    normalized_reference = normalize_for_match(reference)
    if not normalized_reference:
        return False
    return any(normalized_reference in normalize_for_match(c) for c in candidate_texts)


def reference_context_bucket(
    retrieved_sources: Iterable[str],
    retrieved_contents: Iterable[str],
    relevant_documents: Iterable[str],
    reference_contexts: Iterable[str],
) -> str | None:
    """Classify one retrieval's evidence quality for a gold example.

    Shared by `eval/run_eval.py` (the combined/production ranking) and
    `eval/retrieval_attribution.py` (each of dense/BM25/hybrid
    independently) so the A/B/C definition can never quietly diverge
    between the two call sites.

    Parameters
    ----------
    retrieved_sources : Iterable[str]
        A ranking's retrieved chunk `source` paths, in order.
    retrieved_contents : Iterable[str]
        The same ranking's chunk text contents, same order.
    relevant_documents : Iterable[str]
        The gold example's `relevant_documents`.
    reference_contexts : Iterable[str]
        The gold example's `reference_contexts`.

    Returns
    -------
    str | None
        ``None`` if `relevant_documents` is empty (nothing to classify);
        else ``"C"`` (relevant document missed entirely), ``"not_applicable"``
        (document retrieved, but no `reference_contexts` authored to check
        against), ``"B"`` (document retrieved, supporting context missed),
        or ``"A"`` (document retrieved AND supporting context found).
    """
    relevant_documents = list(relevant_documents)
    if not relevant_documents:
        return None
    retrieved_sources = list(retrieved_sources)
    doc_retrieved = any(
        source_matches_relevant(s, rd) for s in retrieved_sources for rd in relevant_documents
    )
    if not doc_retrieved:
        return "C"
    reference_contexts = list(reference_contexts)
    if not reference_contexts:
        return "not_applicable"
    contents = list(retrieved_contents)
    context_hit = all(reference_context_is_supported(ref, contents) for ref in reference_contexts)
    return "A" if context_hit else "B"
