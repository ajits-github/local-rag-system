"""`VisionProvider` backed by a local Ollama vision-capable model."""

from __future__ import annotations

from pathlib import Path

import ollama

from rag.vision.base import VisionProvider

_PROMPT_VERSION = "v1"
_PROMPT = (
    "Describe this image factually for someone who cannot see it. Note the type of "
    "visual (diagram, chart, screenshot, photo) and its key labeled elements, "
    "relationships, or data points. Only describe what is visually present -- do not "
    "guess at unlabeled values, colors, or counts you cannot verify. Keep it to a few "
    "sentences."
)
_ALT_TEXT_PROMPT = (
    "\n\nThe document's own caption/alt text for this image, for grounding only "
    '(do not just repeat it verbatim): "{alt_text}"'
)


class OllamaVisionProvider(VisionProvider):
    """Describes an image via a local Ollama vision model (e.g. `moondream`).

    Offline by construction: `ollama.Client` talks to a native-host
    Ollama server (same one `OllamaLLM` uses for generation), never a
    hosted API, so `config.security.egress_policy` is never involved for
    this provider.
    """

    def __init__(
        self,
        model_name: str = "moondream",
        base_url: str = "http://localhost:11434",
        prompt_version: str = _PROMPT_VERSION,
    ) -> None:
        """Construct the Ollama client used for vision calls.

        Parameters
        ----------
        model_name : str, optional
            Ollama vision-model tag, by default `"moondream"`. Must be
            pulled locally (`ollama pull moondream`) before use;
            never installed automatically.
        base_url : str, optional
            Ollama server URL, by default `"http://localhost:11434"`.
        prompt_version : str, optional
            Identifier for the instruction prompt this provider sends,
            by default `_PROMPT_VERSION`. Exposed as a constructor
            parameter (rather than hardcoded) only so tests can pin a
            distinct value; production callers should leave it at the
            default, which tracks this module's actual `_PROMPT` text.
        """
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name
        self._prompt_version = prompt_version

    @property
    def provider_name(self) -> str:
        """See `VisionProvider.provider_name`."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """See `VisionProvider.model_name`."""
        return self._model_name

    @property
    def prompt_version(self) -> str:
        """See `VisionProvider.prompt_version`."""
        return self._prompt_version

    def describe_image(self, image_path: Path, alt_text: str | None = None) -> str:
        """See `VisionProvider.describe_image`.

        Sends the image's raw bytes via Ollama's `images` message field
        (a local model call, not a hosted upload).
        """
        prompt = _PROMPT
        if alt_text:
            prompt += _ALT_TEXT_PROMPT.format(alt_text=alt_text)
        response = self._client.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt, "images": [image_path.read_bytes()]}],
        )
        return str(response["message"]["content"]).strip()
