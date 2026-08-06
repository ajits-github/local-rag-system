"""Future extension point: image/diagram understanding.

Not wired up anywhere in this codebase yet -- no concrete implementation,
no config block, no factory dispatch, no caller. Ingesting the SVG asset
under data/knowledge_base/assets/ and calling a vision model against it
are both out of scope for the structured-content-ingestion milestone;
this ABC only documents the shape a future implementation would take,
matching this repo's existing base.py-ABC-per-swap-point pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VisionProvider(ABC):
    """Describes an image file in natural language, for ingestion as chunk text."""

    @abstractmethod
    def describe_image(self, image_path: Path, alt_text: str | None = None) -> str:
        """Produce a text description of `image_path`.

        Parameters
        ----------
        image_path : Path
            Path to the image/diagram file.
        alt_text : str | None, optional
            Author-supplied alt text/caption, if any, used as grounding
            context rather than replaced outright, by default None.

        Returns
        -------
        str
            A natural-language description suitable for embedding as
            chunk content. Implementations must not fabricate specifics
            (colors, counts, unlabeled relationships) they cannot verify.
        """
