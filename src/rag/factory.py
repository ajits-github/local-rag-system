"""Single place mapping config `provider` strings to concrete component classes.

Ingestion/retrieval pipelines and the API depend only on the base
interfaces; this module is the one spot that knows about concrete
implementations, so adding a new provider never touches pipeline code.
"""

from __future__ import annotations

from rag.config import AppConfig
from rag.embedders.base import Embedder
from rag.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
from rag.generation.anthropic_llm import AnthropicLLM
from rag.generation.base import LLM
from rag.generation.ollama_llm import OllamaLLM
from rag.generation.openai_llm import OpenAILLM
from rag.rerankers.base import Reranker
from rag.rerankers.cohere import CohereReranker
from rag.rerankers.cross_encoder import CrossEncoderReranker
from rag.rerankers.noop import NoOpReranker
from rag.vectorstore.base import VectorStore
from rag.vectorstore.pgvector import PgVectorStore


def build_embedder(config: AppConfig) -> Embedder:
    """Construct the `Embedder` selected by `config.embedding.provider`.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    Embedder
        The constructed embedder instance.

    Raises
    ------
    ValueError
        If `config.embedding.provider` names an unknown provider.
    """
    if config.embedding.provider == "sentence_transformers":
        return SentenceTransformersEmbedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            normalize=config.embedding.normalize,
        )
    raise ValueError(f"Unknown embedding provider: {config.embedding.provider}")


def build_vectorstore(config: AppConfig) -> VectorStore:
    """Construct the `VectorStore` selected by `config.vectorstore.provider`.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    VectorStore
        The constructed vector store instance.

    Raises
    ------
    ValueError
        If `config.vectorstore.provider` names an unknown provider.
    """
    if config.vectorstore.provider == "pgvector":
        return PgVectorStore(
            dsn=config.database_url(),
            documents_table=config.vectorstore.documents_table,
            chunks_table=config.vectorstore.chunks_table,
            distance_metric=config.vectorstore.distance_metric,
        )
    raise ValueError(f"Unknown vectorstore provider: {config.vectorstore.provider}")


def build_reranker(config: AppConfig) -> Reranker:
    """Construct the `Reranker` selected by `config.reranker.provider`.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    Reranker
        The constructed reranker instance.

    Raises
    ------
    RuntimeError
        If the `cohere` provider is selected but its API key env var is unset.
    ValueError
        If `config.reranker.provider` names an unknown provider.
    """
    provider = config.reranker.provider
    if provider == "none":
        return NoOpReranker()
    if provider == "cross_encoder":
        return CrossEncoderReranker(
            model_name=config.reranker.cross_encoder.model_name,
            device=config.embedding.device,
        )
    if provider == "cohere":
        api_key = config.cohere_api_key()
        if not api_key:
            raise RuntimeError(
                "reranker.provider is 'cohere' but env var "
                f"'{config.reranker.cohere.api_key_env_var}' is not set"
            )
        return CohereReranker(api_key=api_key, model_name=config.reranker.cohere.model_name)
    raise ValueError(f"Unknown reranker provider: {provider}")


def build_llm(config: AppConfig) -> LLM:
    """Construct the `LLM` selected by `config.generation.provider`.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    LLM
        The constructed LLM client instance.

    Raises
    ------
    ValueError
        If `config.generation.provider` names an unknown provider.
    """
    if config.generation.provider == "ollama":
        return OllamaLLM(
            model_name=config.generation.model_name,
            base_url=config.ollama_base_url(),
            temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
        )
    raise ValueError(f"Unknown generation provider: {config.generation.provider}")


def build_judge_llm(config: AppConfig) -> LLM:
    """Construct the RAGAS judge `LLM` selected by `config.judge.provider`.

    Independent of `build_llm`/`config.generation` — the judge is never
    the same model used for generation.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    LLM
        The constructed judge LLM instance.

    Raises
    ------
    RuntimeError
        If a hosted provider (`openai`/`anthropic`) is selected but its
        API key env var is unset.
    ValueError
        If `config.judge.provider` names an unknown provider.
    """
    judge = config.judge
    if judge.provider == "ollama":
        return OllamaLLM(
            model_name=judge.ollama.model_name,
            base_url=config.ollama_base_url(),
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
    if judge.provider == "openai":
        api_key = config.openai_api_key()
        if not api_key:
            raise RuntimeError(
                f"judge.provider is 'openai' but env var '{judge.openai.api_key_env_var}' "
                "is not set"
            )
        return OpenAILLM(
            api_key=api_key,
            model_name=judge.openai.model_name,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
    if judge.provider == "anthropic":
        api_key = config.anthropic_api_key()
        if not api_key:
            raise RuntimeError(
                f"judge.provider is 'anthropic' but env var "
                f"'{judge.anthropic.api_key_env_var}' is not set"
            )
        return AnthropicLLM(
            api_key=api_key,
            model_name=judge.anthropic.model_name,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
    raise ValueError(f"Unknown judge provider: {judge.provider}")
