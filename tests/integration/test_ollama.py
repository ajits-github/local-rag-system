from __future__ import annotations

from rag.generation.ollama_llm import OllamaLLM


def test_ollama_health_check(require_ollama, config):
    llm = OllamaLLM(model_name=config.generation.model_name, base_url=config.ollama_base_url())
    assert llm.health_check() is True


def test_ollama_generate_returns_nonempty_text(require_ollama, config):
    llm = OllamaLLM(
        model_name=config.generation.model_name,
        base_url=config.ollama_base_url(),
        max_tokens=32,
    )
    response = llm.generate("Reply with a single word: hello.")
    assert isinstance(response, str)
    assert len(response.strip()) > 0
