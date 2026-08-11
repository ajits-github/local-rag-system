"""Vision provider: image/diagram understanding for image-derived indexable content.

`config.vision.provider`/`factory.build_vision_provider` wire this ABC in
(see multimodal-ingestion milestone), but the only provider actually
implemented and exercised so far is `"none"` (`build_vision_provider`
returns `None`) -- text-only image handling (alt text/caption, see
chunkers/structured_markdown.py) needs no VisionProvider at all. No
hosted-API-calling concrete subclass exists yet; one is added here only
once a provider/model is chosen and a real (credit-consuming) run is
separately approved. Tests use a local mock subclass, never a class capable
of a real network call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VisionProvider(ABC):
    """Describes an image file in natural language, for ingestion as chunk text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider (e.g. "openai"), for cache/provenance records."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The specific model used, for cache/provenance records."""

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
            Callers (Writer) store this separately from -- never
            overwriting -- the image's original caption/alt-text.
        """
