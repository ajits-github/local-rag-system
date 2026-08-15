"""Lightweight, deterministic prompt-injection phrase detection.

Observability and prompt reinforcement only; never a retrieval gate.
Authorization removes unauthorized content before it reaches the LLM.
This module only flags injection-shaped language so the pipeline can label
suspect sources and feed deterministic safety metrics.

A flagged result is never dropped here. Blocking on this small literal
pattern list would risk false refusals on legitimate content that quotes
or discusses injection attempts.
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
    # Less-literal/obfuscated phrasings a single-pattern list would miss.
    re.compile(r"disregard (the )?(above|prior|previous)", re.I),
    re.compile(r"new instructions\s*:", re.I),
    re.compile(r"forget (everything|all) (you were told|above)", re.I),
    # Letter-spaced/hyphenated obfuscation of "ignore" (e.g. "i-g-n-o-r-e",
    # "i g n o r e") followed eventually by "instructions" -- a targeted,
    # bounded allowance for inter-letter separators, not a general
    # fuzzy-match engine.
    re.compile(r"i[\s\-_.]*g[\s\-_.]*n[\s\-_.]*o[\s\-_.]*r[\s\-_.]*e.{0,20}instructions?", re.I),
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
        semantic judgment.
    """
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
