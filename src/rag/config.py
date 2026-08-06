"""Single entrypoint for loading config/default.yaml into typed settings.

Every infra choice (embedding model, vector backend, chunker, reranker, LLM)
lives in the YAML file as a "provider" field; secrets and connection details
are never hardcoded here — fields named `*_env_var` name an environment
variable that is resolved on demand from the process environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class AppMeta(BaseModel):
    """Top-level application metadata."""

    name: str = "local-rag-system"
    log_level: str = "INFO"


class EmbeddingConfig(BaseModel):
    """Embedding provider selection and model parameters."""

    provider: Literal["sentence_transformers"] = "sentence_transformers"
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    dimension: int


class VectorStoreConfig(BaseModel):
    """Vector store provider selection and table/connection settings."""

    provider: Literal["pgvector"] = "pgvector"
    connection_env_var: str = "DATABASE_URL"
    documents_table: str = "documents"
    chunks_table: str = "chunks"
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"


class ChunkingConfig(BaseModel):
    """Chunker provider selection and chunk size/overlap parameters."""

    provider: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = 500
    chunk_overlap: int = 50


class CrossEncoderConfig(BaseModel):
    """Model settings for the `cross_encoder` reranker provider."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CohereRerankConfig(BaseModel):
    """Model/credential settings for the `cohere` reranker provider."""

    model_name: str = "rerank-english-v3.0"
    api_key_env_var: str = "COHERE_API_KEY"


class RerankerConfig(BaseModel):
    """Reranker provider selection, plus each provider's own settings block."""

    provider: Literal["none", "cross_encoder", "cohere"] = "none"
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    cohere: CohereRerankConfig = Field(default_factory=CohereRerankConfig)


class PromptConfig(BaseModel):
    """Selects and locates the active generation prompt template."""

    id: str = "rag_answer"
    version: str = "v1"
    path: str = "src/rag/prompts/templates/rag_answer_v1.yaml"


class GenerationConfig(BaseModel):
    """LLM provider selection and generation parameters."""

    provider: Literal["ollama"] = "ollama"
    model_name: str = "qwen2.5:1.5b"
    base_url_env_var: str = "OLLAMA_BASE_URL"
    temperature: float = 0.2
    max_tokens: int = 512
    prompt: PromptConfig = Field(default_factory=PromptConfig)


class RetrievalConfig(BaseModel):
    """Result-count tuning for the retrieval pipeline."""

    top_k: int = 5
    rerank_top_n: int = 3


class IngestionConfig(BaseModel):
    """File-type filtering for the ingestion pipeline."""

    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".html", ".htm", ".txt", ".md"]
    )


class AppConfig(BaseModel):
    """Root settings object: the single entrypoint used by the API, ingestion CLI, and eval CLI.

    Loaded once via `load_config` and shared across a process; see that
    function for caching behavior.
    """

    app: AppMeta = Field(default_factory=AppMeta)
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    chunking: ChunkingConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    retrieval: RetrievalConfig
    ingestion: IngestionConfig

    # -- resolved env accessors -------------------------------------------------
    # Kept as methods (not fields) so they always read the *current* process
    # environment rather than a value frozen at load time (handy in tests).

    def database_url(self) -> str:
        """Resolve the Postgres DSN from `vectorstore.connection_env_var`.

        Returns
        -------
        str
            The DSN read from the configured environment variable.

        Raises
        ------
        RuntimeError
            If that environment variable is unset or empty.
        """
        value = os.environ.get(self.vectorstore.connection_env_var)
        if not value:
            raise RuntimeError(
                f"Environment variable '{self.vectorstore.connection_env_var}' is not set. "
                "Set DATABASE_URL, e.g. postgresql://rag:rag@localhost:15987/ragdb"
            )
        return value

    def ollama_base_url(self) -> str:
        """Resolve the Ollama base URL, defaulting to the local instance.

        Returns
        -------
        str
            The URL read from `generation.base_url_env_var`, or
            ``"http://localhost:11434"`` if that variable is unset.
        """
        return os.environ.get(self.generation.base_url_env_var, "http://localhost:11434")

    def cohere_api_key(self) -> str | None:
        """Resolve the Cohere API key from `reranker.cohere.api_key_env_var`.

        Returns
        -------
        str | None
            The API key, or ``None`` if that environment variable is unset.
        """
        return os.environ.get(self.reranker.cohere.api_key_env_var)

    def prompt_template_path(self) -> Path:
        """Resolve `generation.prompt.path`, relative to the repo root if not absolute.

        Returns
        -------
        Path
            The resolved path to the configured prompt template YAML file.
        """
        candidate = Path(self.generation.prompt.path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@lru_cache(maxsize=8)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate config/default.yaml (or an override path).

    Cached by path so the same AppConfig instance is reused across the
    API, ingestion CLI, and eval CLI within a process.

    Parameters
    ----------
    path : str | Path, optional
        Path to a YAML config file, by default `DEFAULT_CONFIG_PATH`.

    Returns
    -------
    AppConfig
        The validated, parsed configuration.
    """
    load_dotenv(override=False)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
