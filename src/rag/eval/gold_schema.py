"""Gold-data schema and path-suffix matching for retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from rag.path_matching import source_matches_relevant as _shared_source_matches_relevant


class GoldExample(BaseModel):
    """One row of a gold JSONL file (e.g. data/eval/techfusion_gold.jsonl).

    `relevant_documents` are paths *relative to wherever the knowledge base
    root is* (e.g. "knowledge_base/security/access-control-policy.md"), not
    document_id UUIDs — ids are assigned at ingestion time and can't be
    known ahead of authoring gold data. Matching against a retrieved
    chunk's stored `source` is therefore done by path-suffix (see
    `source_matches_relevant`), which works regardless of what root path a
    particular ingestion run used — nothing here hardcodes "data/" or
    "knowledge_base/".

    `reference_contexts`/`reference_visual_contexts` are evaluation-only
    ground truth, never indexed and never passed to ingestion/retrieval/
    generation -- eval/*.py is the only code that reads them. The former is
    textual evidence expected to be a verbatim excerpt from the source
    corpus (used by eval/run_eval.py's supporting-context-hit check); the
    latter is evaluation ground truth for facts that are visually present in
    an image but intentionally absent from its caption/surrounding text
    (e.g. an exact chart value), used only for reporting, never for
    indexing or retrieval matching. `relevant_images` names the image
    asset(s) associated with a question; `requires_vision` marks questions
    that cannot legitimately be answered from text/caption context alone.
    """

    question: str
    expected_answer: str | None = None
    relevant_documents: list[str] = Field(default_factory=list)
    question_type: str | None = None
    difficulty: str | None = None
    unanswerable: bool = False
    # Multimodal/relationship-aware milestone fields (all optional/defaulted
    # so pre-existing gold files without them, e.g. techfusion_gold_old.jsonl
    # and sample_gold.jsonl, keep parsing unchanged). `content_type` here is
    # an *authored* ground-truth question category (e.g. "image_only",
    # "relationship_aware") -- distinct from eval/content_type.py's
    # chunker-derived document buckets; see CLAUDE.md/PROJECT_JOURNAL.md for
    # why the two aren't conflated.
    content_type: str | None = None
    reference_contexts: list[str] = Field(default_factory=list)
    reference_visual_contexts: list[str] = Field(default_factory=list)
    relevant_images: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    requires_vision: bool = False
    requires_relationship_expansion: bool = False
    # Safety/freshness milestone fields (all optional/defaulted so
    # pre-existing gold files without them, e.g. techfusion_gold_without_safety.jsonl,
    # keep parsing unchanged). safety_category is None for every plain-text
    # row; user_tenant/user_roles/query_as_of are the caller identity/date
    # eval/run_eval.py builds an AuthorizationContext from (see
    # retrieval/authorization.py). allowed_documents/forbidden_documents are
    # ground truth for the unauthorized_retrieval_rate/cross_tenant_leakage_rate
    # metrics; expected_behavior drives refusal_accuracy/false_refusal_rate.
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

    E.g. gold's "knowledge_base/security/x.md" matches a stored source of
    "data/knowledge_base/security/x.md" regardless of the "data/" root,
    without either side needing to hardcode that root. Re-exported from
    `rag.path_matching` (moved there so `rag.retrieval.freshness` can reuse
    the identical rule without an eval->retrieval-adjacent import — see that
    module's docstring).

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
    source corpus (see `GoldExample`'s docstring), so this only smooths
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
    retrieved chunk) -- same matching rule, different candidate sets.

    **Limitation** (documented, not hidden): this is substring containment
    after whitespace normalization, not semantic similarity. It is valid
    because `reference_contexts` entries are authored as verbatim excerpts
    (confirmed against real gold rows: exact JSON blocks, exact backtick
    commands, exact caption sentences including their `*asterisks*`), but
    it is brittle to any *other* formatting drift between the gold
    author's copy and the actual text being checked against -- e.g. a
    reference spanning a table row-group boundary that got split across
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
