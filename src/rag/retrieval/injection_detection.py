"""Lightweight, deterministic prompt-injection phrase detection.

Observability/prompt-reinforcement only -- **never a gate**. Authorization
(`vectorstore.pgvector._build_authorization_where_clause`) is what actually
removes unauthorized content before it reaches the LLM; this module only
flags a query or a retrieved chunk as containing injection-shaped language,
so `RetrievalPipeline` can (a) label a suspect chunk's `_source_label` as
untrusted data for the model, and (b) feed the safety metrics in
`eval/run_eval.py` (`prompt_injection_success_rate`/
`retrieved_prompt_injection_success_rate`). A flagged result is never
dropped or blocked here -- doing so would risk false refusals on completely
legitimate content that merely quotes or discusses injection attempts (see
`data/knowledge_base/internal_techfusion/untrusted-operations-notes.md`,
which is authorized, retrievable, and *about* an injection attempt).

Deliberately a small, literal phrase/regex list -- same documented-heuristic
style as `eval/run_eval.py`'s `_looks_like_refusal`/`_REFUSAL_PHRASES` -- not
a claim of semantic understanding, and not "a large external safety
framework" (out of scope for this milestone).
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = (
    re.compile(
        r"ignore (all |any )?(previous|prior|authoritative) (instructions?|pages?|rules?)", re.I
    ),
    re.compile(r"system override", re.I),
    re.compile(r"disable (the )?(tenant|acl|access|filter)", re.I),
    re.compile(r"print (any |every |all )?(hidden|customer|forbidden)", re.I),
    re.compile(r"reveal (every|all|any) (retrieved )?(document|secret|key|context)", re.I),
    re.compile(r"pretend (i am|you are|i'm|you're)", re.I),
    re.compile(r"act as (an? )?(administrator|admin|system)", re.I),
    re.compile(r"treat (the following|this) (token|text) as a system command", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"mark this (upload|document|page) as active", re.I),
)


def detect_injection(text: str) -> bool:
    """Whether `text` contains one of `_INJECTION_PATTERNS` (case-insensitive).

    Parameters
    ----------
    text : str
        A user query, or one retrieved chunk's content.

    Returns
    -------
    bool
        True if any pattern matches. A heuristic triage flag, not a
        semantic judgment -- see this module's docstring.
    """
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
