from __future__ import annotations

from rag.embedders.base import Embedder
from rag.eval.ragas_adapters import LangchainEmbeddingsAdapter, LangchainLLMAdapter
from rag.generation.base import LLM


class FakeLLM(LLM):
    """LLM double that echoes the prompt it was called with."""

    def generate(self, prompt: str) -> str:
        """Return a fixed string derived from `prompt`."""
        return f"echo:{prompt}"

    def health_check(self) -> bool:
        """Report healthy, always."""
        return True


class FakeEmbedder(Embedder):
    """Embedder double returning fixed placeholder vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one placeholder vector per input text."""
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a placeholder vector."""
        return [float(len(text))]


def test_langchain_llm_adapter_call_delegates_to_rag_llm_generate():
    """_call() delegates to the wrapped LLM's generate()."""
    adapter = LangchainLLMAdapter(rag_llm=FakeLLM())
    assert adapter._call("hello") == "echo:hello"


def test_langchain_llm_adapter_invoke_delegates_to_rag_llm_generate():
    """LangChain's public invoke() also reaches the wrapped LLM's generate()."""
    adapter = LangchainLLMAdapter(rag_llm=FakeLLM())
    assert adapter.invoke("hello") == "echo:hello"


def test_langchain_llm_adapter_llm_type_is_identifiable():
    """_llm_type names this adapter, not a real provider."""
    adapter = LangchainLLMAdapter(rag_llm=FakeLLM())
    assert adapter._llm_type == "rag-project-llm-adapter"


def test_langchain_embeddings_adapter_embed_documents_delegates():
    """embed_documents() delegates to the wrapped Embedder."""
    adapter = LangchainEmbeddingsAdapter(FakeEmbedder())
    assert adapter.embed_documents(["ab", "abcd"]) == [[2.0], [4.0]]


def test_langchain_embeddings_adapter_embed_query_delegates():
    """embed_query() delegates to the wrapped Embedder."""
    adapter = LangchainEmbeddingsAdapter(FakeEmbedder())
    assert adapter.embed_query("abc") == [3.0]
