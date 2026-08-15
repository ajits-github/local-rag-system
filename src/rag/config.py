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


class StructuredMarkdownConfig(BaseModel):
    """Tunables specific to the `structured_markdown` chunking provider."""

    table_row_group_size: int = 20
    max_atomic_block_chars: int = 2000


class ChunkingConfig(BaseModel):
    """Chunker provider selection and chunk size/overlap parameters."""

    provider: Literal["recursive_character", "structured_markdown"] = "structured_markdown"
    chunk_size: int = 500
    chunk_overlap: int = 50
    structured_markdown: StructuredMarkdownConfig = Field(default_factory=StructuredMarkdownConfig)


class CrossEncoderConfig(BaseModel):
    """Model settings for the `cross_encoder` reranker provider."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CohereRerankConfig(BaseModel):
    """Model/credential settings for the `cohere` reranker provider."""

    model_name: str = "rerank-english-v3.0"
    api_key_env_var: str = "COHERE_API_KEY"


class RerankerConfig(BaseModel):
    """Reranker provider selection, plus each provider's own settings block.

    `top_n` has reranker semantics only: how many candidates a real
    reranker (`cross_encoder` or `cohere`) retains after rescoring. It is
    inert when `provider == "none"` because `NoOpReranker` is an identity.
    The final number of chunks that reach generation is controlled
    independently by `retrieval.generation_context_top_n`.
    """

    provider: Literal["none", "cross_encoder", "cohere"] = "none"
    top_n: int = 5
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    cohere: CohereRerankConfig = Field(default_factory=CohereRerankConfig)


class PromptConfig(BaseModel):
    """Selects and locates the active generation prompt template."""

    id: str = "rag_answer"
    version: str = "v1"
    path: str = "src/rag/prompts/templates/rag_answer_v1.yaml"


class GenerationConfig(BaseModel):
    """LLM provider selection and generation parameters.

    `seed` is `None` by default, preserving non-deterministic sampling.
    Ollama's generation seed makes repeated calls reproducible only for
    the same model file, quantization, runtime, and options; it is not a
    cross-machine determinism guarantee. Use it with `temperature: 0` for
    a reproducible local baseline.
    """

    provider: Literal["ollama"] = "ollama"
    model_name: str = "qwen2.5:1.5b"
    base_url_env_var: str = "OLLAMA_BASE_URL"
    temperature: float = 0.2
    max_tokens: int = 512
    seed: int | None = None
    prompt: PromptConfig = Field(default_factory=PromptConfig)


class OpenAIJudgeConfig(BaseModel):
    """Model/credential settings for the `openai` judge provider."""

    model_name: str = "gpt-4o-mini"
    api_key_env_var: str = "OPENAI_API_KEY"


class AnthropicJudgeConfig(BaseModel):
    """Model/credential settings for the `anthropic` judge provider."""

    model_name: str = "claude-haiku-4-5-20251001"
    api_key_env_var: str = "ANTHROPIC_API_KEY"


class OllamaJudgeConfig(BaseModel):
    """Model settings for the `ollama` (local, experimental) judge provider."""

    model_name: str = "llama3.1:8b"


class JudgeConfig(BaseModel):
    """RAGAS judge LLM provider selection, independent of `generation`.

    Never defaults to the generation models (qwen2.5:1.5b/3b) — hosted
    providers are the primary supported path; `ollama` is offered for
    free local experimentation only, with a different, non-default model.
    """

    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    temperature: float = 0.0
    max_tokens: int = 1024
    cache_enabled: bool = True
    cache_dir: str = ".cache/ragas"
    openai: OpenAIJudgeConfig = Field(default_factory=OpenAIJudgeConfig)
    anthropic: AnthropicJudgeConfig = Field(default_factory=AnthropicJudgeConfig)
    ollama: OllamaJudgeConfig = Field(default_factory=OllamaJudgeConfig)


class MLflowConfig(BaseModel):
    """Experiment-tracking sink: every `scripts/record_experiment.py` run also logs to MLflow.

    Local SQLite backend by default (`sqlite:///mlflow.db`), so no MLflow
    server is required for local experiment records. Point `tracking_uri`
    at a hosted MLflow server when a shared tracker is needed.
    """

    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "local-rag-system"


class HybridRetrievalConfig(BaseModel):
    """Tunables specific to the `hybrid` retrieval provider."""

    rrf_k: int = 60


class RelationshipExpansionConfig(BaseModel):
    """Post-rerank context-expansion tunables.

    Parameters
    ----------
    enabled : bool
        When `False` (the default), a no-op.
    include_parent : bool
        Append each result's nearest preceding prose chunk, if any.
    include_neighbors : bool
        Append each result's immediate chunk-index neighbors in the
        same section.
    max_related_elements : int
        Maximum expanded chunks appended per originating result.
    """

    enabled: bool = False
    include_parent: bool = True
    include_neighbors: bool = True
    max_related_elements: int = 3


class RetrievalConfig(BaseModel):
    """Retrieval provider selection and result-count tuning.

    Parameters
    ----------
    provider : {"dense", "hybrid"}
        Retrieval strategy. `"hybrid"` adds BM25 search fused with RRF.
    candidate_k : int
        Candidate pool depth fetched from the vector store before
        reranking.
    generation_context_top_n : int
        Ranked primary chunks kept for generation, applied after the
        optional rerank step. See docs/architecture.md's "Retrieval
        Cutoff Semantics" section for how this relates to
        `reranker.top_n`.
    hybrid : HybridRetrievalConfig
        Hybrid-retrieval-specific settings.
    relationship_expansion : RelationshipExpansionConfig
        Post-rerank context-expansion settings.
    """

    provider: Literal["dense", "hybrid"] = "dense"
    candidate_k: int = 5
    generation_context_top_n: int = 3
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    relationship_expansion: RelationshipExpansionConfig = Field(
        default_factory=RelationshipExpansionConfig
    )


class IngestionConfig(BaseModel):
    """File-type filtering for the ingestion pipeline."""

    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".html", ".htm", ".txt", ".md"]
    )


class AuthorizationConfig(BaseModel):
    """Retrieval-time tenant/role/freshness enforcement tunables.

    Parameters
    ----------
    enabled : bool
        When `False` (the default), authorization is a no-op: chunks
        with no `tenant_id` (the entire pre-existing corpus) pass
        unconditionally either way.
    cross_tenant_support_roles : list[str]
        Role names allowed to cross a tenant boundary, but only when
        that same role also appears in the target document's own
        `allowed_roles`.
    """

    enabled: bool = False
    cross_tenant_support_roles: list[str] = Field(default_factory=lambda: ["techfusion_support"])


class FieldRedactionConfig(BaseModel):
    """Field-level sensitive-value redaction tunable.

    Parameters
    ----------
    enabled : bool
        When `False` (the default), a no-op. Independent of
        `AuthorizationConfig.enabled` — document ACL and field
        disclosure are separate controls.

    Notes
    -----
    When enabled, a missing caller identity is treated as zero roles
    (fail-closed), never as unrestricted.
    """

    enabled: bool = False


class JWTConfig(BaseModel):
    """JWT verification tunables for `AuthConfig`.

    Parameters
    ----------
    algorithm : {"HS256", "RS256", "ES256"}
        Signing algorithm. `HS256` (shared-secret) is the default for
        this single-instance, offline-first system; `RS256`/`ES256`
        support fronting a real identity provider later.
    secret_env_var : str
        Environment variable naming the HS256 shared secret.
    public_key_path_env_var : str
        Environment variable naming a PEM file path, for RS256/ES256.
    issuer : str or None
        Expected `iss` claim; checked only when set.
    audience : str or None
        Expected `aud` claim; checked only when set.
    leeway_seconds : int
        Clock-skew tolerance for expiration checks.
    required_claims : list[str]
        Claims that must be present; a token missing any is rejected as
        malformed regardless of signature validity.
    """

    algorithm: Literal["HS256", "RS256", "ES256"] = "HS256"
    secret_env_var: str = "JWT_HS256_SECRET"
    public_key_path_env_var: str = "JWT_PUBLIC_KEY_PATH"
    issuer: str | None = None
    audience: str | None = None
    leeway_seconds: int = 30
    required_claims: list[str] = Field(default_factory=lambda: ["sub", "tenant_id", "roles"])


class AuthConfig(BaseModel):
    """API-boundary JWT authentication tunables.

    Parameters
    ----------
    enabled : bool
        When `False` (the default), `POST /query`/`POST /ingest` trust
        caller-supplied `tenant_id`/`roles` directly. When `True`, a
        verified bearer token is required and body-supplied
        `tenant_id`/`roles` are ignored for authorization.
    insecure_dev_mode : bool
        Only relevant when `enabled=True`. Allows requests with no
        `Authorization` header at all; an invalid, expired, or malformed
        token is always rejected regardless of this flag.
    ingest_roles : list[str]
        Roles allowed to call `POST /ingest` when `enabled=True`.
    jwt : JWTConfig
        JWT verification settings.
    """

    enabled: bool = False
    insecure_dev_mode: bool = False
    ingest_roles: list[str] = Field(
        default_factory=lambda: ["security_admin", "system_admin", "techfusion_support"]
    )
    jwt: JWTConfig = Field(default_factory=JWTConfig)


class DoSLimitsConfig(BaseModel):
    """Bounded request-size validation for `POST /query`/`POST /ingest`.

    Parameters
    ----------
    max_query_length : int
        Maximum allowed `query` string length, in characters.
    max_top_k : int
        Maximum allowed `top_k` value.
    max_filters_bytes : int
        Maximum allowed serialized size of the `filters` dict.
    max_upload_bytes : int
        Maximum allowed upload size for `POST /ingest`.

    Notes
    -----
    Enforced as router-level checks rather than Pydantic field
    constraints, since the bounds must read from runtime config.
    """

    max_query_length: int = 2000
    max_top_k: int = 20
    max_filters_bytes: int = 4096
    max_upload_bytes: int = 26_214_400  # 25 MiB


class RateLimitConfig(BaseModel):
    """Per-caller request-rate limiting (`slowapi`, in-memory backend).

    Parameters
    ----------
    enabled : bool
        When `False` (the default), a no-op.
    requests_per_minute : int
        Allowed requests per minute, per bucket.
    key : {"tenant", "ip"}
        Bucketing key. `"tenant"` uses the verified identity's
        `tenant_id` when present, falling back to client IP.

    Notes
    -----
    In-memory state is process-local; with more than one API replica,
    each enforces its own independent limit.
    """

    enabled: bool = False
    requests_per_minute: int = 60
    key: Literal["tenant", "ip"] = "tenant"


class EgressPolicyConfig(BaseModel):
    """Provider-egress policy gate for hosted-judge (RAGAS) calls only.

    Parameters
    ----------
    enabled : bool
        When `False` (the default), a no-op — retrieved context reaches
        the configured hosted judge unchecked.
    block_unredacted_sensitive_fields : bool
        Block a source whose tagged sensitive fields were not fully
        redacted for this retrieval.
    require_authoritative_trust : bool
        Block any source whose `trust_level` is not `"authoritative"`.
    blocked_tenant_ids : list[str]
        Tenants whose content may never egress.
    classification_policy : dict[str, bool]
        Maps a document `classification` to whether it may egress. A
        classification with no entry is blocked (fail-closed); missing
        classification is treated as allowed, matching `"internal"`.
    """

    enabled: bool = False
    block_unredacted_sensitive_fields: bool = True
    require_authoritative_trust: bool = False
    blocked_tenant_ids: list[str] = Field(default_factory=list)
    classification_policy: dict[str, bool] = Field(
        default_factory=lambda: {
            "public": True,
            "internal": True,
            "confidential": False,
            "restricted": False,
        }
    )


class SecurityConfig(BaseModel):
    """Root config block for authorization, redaction, authentication, and hardening controls."""

    authorization: AuthorizationConfig = Field(default_factory=AuthorizationConfig)
    field_redaction: FieldRedactionConfig = Field(default_factory=FieldRedactionConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    dos_limits: DoSLimitsConfig = Field(default_factory=DoSLimitsConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    egress_policy: EgressPolicyConfig = Field(default_factory=EgressPolicyConfig)


class VisionConfig(BaseModel):
    """Vision provider selection for image-derived indexable content.

    `"none"` (the default, and the only provider actually exercised so
    far) means no image bytes are ever read and no VisionProvider is
    constructed. Text-only image handling, based on alt text and
    captions, works with this config untouched.
    Concrete hosted providers are added to `factory.build_vision_provider`
    only once explicitly approved for a real (credit-consuming) run; no
    hosted provider name is declared here yet.
    """

    provider: Literal["none"] = "none"


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
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    retrieval: RetrievalConfig
    ingestion: IngestionConfig
    vision: VisionConfig = Field(default_factory=VisionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Resolved environment accessors.
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

    def openai_api_key(self) -> str | None:
        """Resolve the OpenAI API key from `judge.openai.api_key_env_var`.

        Returns
        -------
        str | None
            The API key, or ``None`` if that environment variable is unset.
        """
        return os.environ.get(self.judge.openai.api_key_env_var)

    def anthropic_api_key(self) -> str | None:
        """Resolve the Anthropic API key from `judge.anthropic.api_key_env_var`.

        Returns
        -------
        str | None
            The API key, or ``None`` if that environment variable is unset.
        """
        return os.environ.get(self.judge.anthropic.api_key_env_var)

    def jwt_signing_key(self) -> str | bytes:
        """Resolve the JWT verification key per `security.auth.jwt.algorithm`.

        Returns
        -------
        str | bytes
            The HS256 shared secret (from `jwt.secret_env_var`), or the
            PEM-encoded public key bytes read from the file named by
            `jwt.public_key_path_env_var` for RS256/ES256.

        Raises
        ------
        RuntimeError
            If the relevant environment variable is unset or empty, or
            (for RS256/ES256) the PEM file it names does not exist.
        """
        jwt_config = self.security.auth.jwt
        if jwt_config.algorithm == "HS256":
            value = os.environ.get(jwt_config.secret_env_var)
            if not value:
                raise RuntimeError(
                    f"Environment variable '{jwt_config.secret_env_var}' is not set. "
                    "Set it to the HS256 shared signing secret."
                )
            return value
        path_value = os.environ.get(jwt_config.public_key_path_env_var)
        if not path_value:
            raise RuntimeError(
                f"Environment variable '{jwt_config.public_key_path_env_var}' is not set. "
                f"Set it to a PEM public key file path for {jwt_config.algorithm}."
            )
        key_path = Path(path_value)
        if not key_path.is_file():
            raise RuntimeError(f"JWT public key file not found: {key_path}")
        return key_path.read_bytes()

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
