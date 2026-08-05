"""Default `Cleaner`: whitespace/control-char normalization only."""

from __future__ import annotations

import re

from rag.cleaners.base import Cleaner

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+\n")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class DefaultCleaner(Cleaner):
    """Whitespace/control-char normalization; no content rewriting."""

    def clean(self, text: str) -> str:
        """Strip control characters, trailing whitespace, and excess blank lines.

        Parameters
        ----------
        text : str
            Raw text to normalize.

        Returns
        -------
        str
            Normalized, stripped text.
        """
        text = _CONTROL_CHARS.sub("", text)
        text = _TRAILING_WHITESPACE.sub("\n", text)
        text = _MULTI_BLANK_LINES.sub("\n\n", text)
        return text.strip()
