from __future__ import annotations

from pathlib import Path

from rag.vision.ollama_vision import OllamaVisionProvider


class _FakeOllamaClient:
    """Stand-in for `ollama.Client`, mocked at the class boundary per project convention."""

    def __init__(self, response: dict) -> None:
        """Store the fixed `chat()` response this double will return."""
        self._response = response
        self.last_messages: list[dict] | None = None
        self.last_model: str | None = None

    def chat(self, model: str, messages: list[dict]) -> dict:
        """Return the fixed response, recording `model`/`messages` for assertions."""
        self.last_model = model
        self.last_messages = messages
        return self._response


def _make_provider(
    monkeypatch, response: dict, **kwargs
) -> tuple[OllamaVisionProvider, _FakeOllamaClient]:
    """Build an OllamaVisionProvider whose internal ollama.Client is replaced with a fake."""
    provider = OllamaVisionProvider(model_name="moondream", **kwargs)
    fake = _FakeOllamaClient(response)
    monkeypatch.setattr(provider, "_client", fake)
    return provider, fake


def test_describe_image_returns_stripped_model_response(monkeypatch, tmp_path: Path):
    """describe_image returns the chat response content, stripped of surrounding whitespace."""
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png-bytes")
    provider, _ = _make_provider(
        monkeypatch, {"message": {"content": "  A diagram of a pipeline.  "}}
    )

    description = provider.describe_image(image_path)

    assert description == "A diagram of a pipeline."


def test_describe_image_sends_raw_image_bytes(monkeypatch, tmp_path: Path):
    """The image file's raw bytes are passed via the message's images field, not a file path."""
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png-bytes")
    provider, fake = _make_provider(monkeypatch, {"message": {"content": "A diagram."}})

    provider.describe_image(image_path)

    assert fake.last_model == "moondream"
    sent_images = fake.last_messages[0]["images"]
    assert sent_images == [b"fake-png-bytes"]


def test_describe_image_includes_alt_text_as_grounding_context(monkeypatch, tmp_path: Path):
    """When alt_text is given, it's appended to the prompt, not used as a substitute answer."""
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png-bytes")
    provider, fake = _make_provider(monkeypatch, {"message": {"content": "A diagram."}})

    provider.describe_image(image_path, alt_text="Architecture overview")

    prompt = fake.last_messages[0]["content"]
    assert "Architecture overview" in prompt


def test_describe_image_omits_alt_text_block_when_none(monkeypatch, tmp_path: Path):
    """No alt_text. the prompt doesn't reference any caption/alt-text grounding at all."""
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png-bytes")
    provider, fake = _make_provider(monkeypatch, {"message": {"content": "A diagram."}})

    provider.describe_image(image_path, alt_text=None)

    prompt = fake.last_messages[0]["content"]
    assert "caption/alt text" not in prompt


def test_provider_identity_fields():
    """provider_name/model_name/prompt_version are stable, non-network-calling properties."""
    provider = OllamaVisionProvider(model_name="moondream", prompt_version="v2")

    assert provider.provider_name == "ollama"
    assert provider.model_name == "moondream"
    assert provider.prompt_version == "v2"
