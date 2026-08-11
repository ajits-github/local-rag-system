"""Gold-data schema and path-suffix matching for retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field


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


def _path_parts(path: str) -> tuple[str, ...]:
    """Split a possibly-backslash path into POSIX-style path segments."""
    return PurePosixPath(path.replace("\\", "/")).parts


def source_matches_relevant(retrieved_source: str, relevant_path: str) -> bool:
    """Check whether `relevant_path`'s segments trail `retrieved_source`'s segments.

    E.g. gold's "knowledge_base/security/x.md" matches a stored source of
    "data/knowledge_base/security/x.md" regardless of the "data/" root,
    without either side needing to hardcode that root.

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
    retrieved_parts = _path_parts(retrieved_source)
    relevant_parts = _path_parts(relevant_path)
    if not relevant_parts or len(relevant_parts) > len(retrieved_parts):
        return False
    return retrieved_parts[-len(relevant_parts) :] == relevant_parts


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
