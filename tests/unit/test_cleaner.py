from __future__ import annotations

from rag.cleaners.default_cleaner import DefaultCleaner


def test_default_cleaner_collapses_excess_blank_lines():
    """Runs of 3+ blank lines are collapsed to a single blank line."""
    cleaned = DefaultCleaner().clean("line one\n\n\n\n\nline two")
    assert cleaned == "line one\n\nline two"


def test_default_cleaner_strips_trailing_whitespace_and_control_chars():
    """Trailing whitespace and control characters are removed."""
    cleaned = DefaultCleaner().clean("line one   \nline two\x00\x01\n  ")
    assert "\x00" not in cleaned
    assert cleaned == "line one\nline two"
