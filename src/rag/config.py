"""Single entrypoint for loading config/default.yaml into typed settings.

Every infra choice (embedding model, vector backend, chunker, reranker, LLM)
lives in the YAML file as a "provider" field; secrets and connection details
are never hardcoded here. Fields named `*_env_var` name an environment
variable that is resolved on demand from the process environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

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
    """Vector store provider selection and table/connection settings.

    Attributes
    ----------
    provider : {"pgvector"}
        The vector store backend.
    connection_env_var : str
        Name of the environment variable holding the Postgres DSN.
    documents_table, chunks_table : str
        Table names.
    distance_metric : {"cosine", "l2", "inner_product"}
        Distance metric used for vector similarity search.
    minconn, maxconn : int
        `psycopg2.pool.ThreadedConnectionPool` bounds for `PgVectorStore`'s
        own pool. `FeedbackStore`'s pool size is configured independently
        on `FeedbackConfig`, since that pool is deliberately small and
        separate. Raised slightly above the previous hardcoded defaults
        (`minconn=1`, `maxconn=5`) to give a small burst of concurrent
        requests more headroom before exhausting the pool.
    connect_timeout_seconds : int
        `psycopg2.connect`'s `connect_timeout`, in seconds: how long a
        pooled connection attempt waits before giving up, so a genuinely
        unreachable Postgres instance fails fast instead of hanging a
        request indefinitely.
    statement_timeout_ms : int
        Postgres's own `statement_timeout` (via `options='-c
        statement_timeout=<ms>'` on every new connection), in
        milliseconds: how long the server lets a single query run before
        cancelling it, so a pathological/runaway query can't hold a pooled
        connection (and a request) open forever.
    """

    provider: Literal["pgvector"] = "pgvector"
    connection_env_var: str = "DATABASE_URL"
    documents_table: str = "documents"
    chunks_table: str = "chunks"
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    minconn: int = 1
    maxconn: int = 10
    connect_timeout_seconds: int = 5
    statement_timeout_ms: int = 30_000


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

    Never defaults to the generation models (qwen2.5:1.5b/3b). Hosted
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


class MetricsConfig(BaseModel):
    """Prometheus `/metrics` endpoint tunable.

    Attributes
    ----------
    enabled : bool
        When `True` (the default), `/metrics` serves Prometheus text
        exposition and every request/agent metric is recorded. Harmless
        on by default: purely in-process counters/histograms, no
        external service required for this to be a no-op in practice
        unless something actually scrapes the endpoint. Same
        "default-on, no external dependency" precedent as `MLflowConfig`.
    path : str
        The mount path for the metrics endpoint.
    """

    enabled: bool = True
    path: str = "/metrics"


class TracingConfig(BaseModel):
    """OpenTelemetry distributed-tracing tunable.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a true no-op: the OpenTelemetry API's
        own built-in no-op `TracerProvider` stays in place, so every
        `start_span` call in the codebase is free. `True` requires an
        OTLP endpoint actually listening (see `AppConfig.otlp_endpoint`,
        resolved from `OTEL_EXPORTER_OTLP_ENDPOINT`). Default off so a
        fresh checkout never produces background exporter connection
        errors with nothing running to receive them.
    service_name : str
        The `service.name` resource attribute on every exported span.
    sample_ratio : float
        Fraction of traces sampled, in `[0.0, 1.0]`. `1.0` (the default)
        is appropriate for local development's low request volume.
    """

    enabled: bool = False
    service_name: str = "local-rag-system"
    sample_ratio: float = 1.0


class LiveEventsConfig(BaseModel):
    """`POST /agent/query/stream` (SSE) tunable.

    Attributes
    ----------
    enabled : bool
        When `True` (the default), the streaming endpoint is mounted.
        Adds a route only. No external dependency, no cost unless a
        caller actually opens a stream.
    """

    enabled: bool = True


class ObservabilityConfig(BaseModel):
    """Root config block for operational telemetry: metrics, tracing, live events.

    Independent of `MLflowConfig`: operational telemetry and experiment
    tracking are separate concerns, and neither replaces the other.
    """

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    live_events: LiveEventsConfig = Field(default_factory=LiveEventsConfig)


class HybridRetrievalConfig(BaseModel):
    """Tunables specific to the `hybrid` retrieval provider."""

    rrf_k: int = 60


class RelationshipExpansionConfig(BaseModel):
    """Post-rerank context-expansion tunables.

    Attributes
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

    Attributes
    ----------
    provider : {"dense", "hybrid"}
        Retrieval strategy. `"hybrid"` adds BM25 search fused with RRF.
    candidate_k : int
        Candidate pool depth fetched from the vector store before
        reranking.
    generation_context_top_n : int
        Ranked primary chunks kept for generation, applied uniformly
        after the optional rerank step. Independent of `reranker.top_n`
        (how many a real reranker keeps after rescoring); this cutoff
        applies even when `reranker.provider == "none"`.
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


class LayoutParsingConfig(BaseModel):
    """PDF/DOCX structural-extraction tunables (headings, tables, images, pages).

    `pdf_parser` is the swap point for this stage: currently only
    `"pdfplumber"` is implemented (MIT-licensed; used for word-level font
    sizes, table extraction, and page numbers), but `loaders/pdf_loader.py`
    reads this field rather than hardcoding the library, so a future
    parser can be added without touching the chunker or the rest of
    ingestion.
    """

    pdf_parser: Literal["pdfplumber"] = "pdfplumber"
    heading_font_size_ratio: float = 1.15
    title_font_size_ratio: float = 1.8
    max_heading_words: int = 12


class IngestionConfig(BaseModel):
    """File-type filtering and layout-parsing tunables for the ingestion pipeline."""

    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".html", ".htm", ".txt", ".md"]
    )
    layout_parsing: LayoutParsingConfig = Field(default_factory=LayoutParsingConfig)


class AuthorizationConfig(BaseModel):
    """Retrieval-time tenant/role/freshness enforcement tunables.

    Attributes
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

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a no-op. Independent of
        `AuthorizationConfig.enabled`. Document ACL and field
        disclosure are separate controls.

    Notes
    -----
    When enabled, a missing caller identity is treated as zero roles
    (fail-closed), never as unrestricted.
    """

    enabled: bool = False


class JWTConfig(BaseModel):
    """JWT verification tunables for `AuthConfig`.

    Attributes
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
        Claims that must be present; a token missing any is rejected with
        reason `missing_claim` (see `AuthFailureReason`), independent of
        signature validity.
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

    Attributes
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

    Attributes
    ----------
    max_query_length : int
        Maximum allowed `query` string length, in characters.
    max_top_k : int
        Maximum allowed `top_k` value.
    max_filters_bytes : int
        Maximum allowed serialized size of the `filters` dict.
    max_upload_bytes : int
        Maximum allowed upload size for `POST /ingest`.
    max_history_turns : int
        Maximum allowed length of a request's `conversation_history`
        list (see `rag.context.history`). Bounds the request payload
        itself, independent of `context.history.max_turns`, which
        bounds how many of those turns actually reach the prompt.
    max_history_turn_chars : int
        Maximum allowed character length of one `conversation_history`
        turn's `content`.

    Notes
    -----
    Enforced as router-level checks rather than Pydantic field
    constraints, since the bounds must read from runtime config.
    """

    max_query_length: int = 2000
    max_top_k: int = 20
    max_filters_bytes: int = 4096
    max_upload_bytes: int = 26_214_400  # 25 MiB
    max_history_turns: int = 20
    max_history_turn_chars: int = 4000


class RateLimitConfig(BaseModel):
    """Per-caller request-rate limiting (`slowapi`, in-memory backend).

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a no-op.
    requests_per_minute : int
        Allowed requests per minute, per bucket.
    key : {"tenant", "ip"}
        Bucketing key, read by `api/deps.py:_rate_limit_key`. `"tenant"`
        (the default) buckets by the verified identity's `tenant_id` when
        present, falling back to client IP; `"ip"` always buckets by
        client IP, even when a verified identity is present.

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

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a no-op. Retrieved context reaches
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
        classification with no entry is blocked (fail-closed).
    allow_unclassified : bool
        Whether a source with no `classification` may egress. `True` keeps
        the pre-governance corpus usable; production-shaped configs set it
        `False` so a document ingested without front matter fails closed.
    """

    enabled: bool = False
    block_unredacted_sensitive_fields: bool = True
    require_authoritative_trust: bool = False
    blocked_tenant_ids: list[str] = Field(default_factory=list)
    allow_unclassified: bool = True
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


class AgentConfig(BaseModel):
    """Bounded agentic-RAG workflow tunables, layered above the classic retrieval pipeline.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), `POST /agent/query` still exists but
        `classify_query` always routes to `classic_rag`. A coarse
        kill-switch matching `AuthorizationConfig`/`FieldRedactionConfig`'s
        convention.
    max_agent_steps : int
        Hard ceiling on total graph node executions per request.
    max_retrieval_attempts : int
        Hard ceiling on reformulate-and-retry loop iterations.
    max_tool_calls : int
        Hard ceiling on total tool dispatches (search + document + latest
        + related context combined).
    max_json_parse_retries : int
        Bounded reparse-nudge attempts when an LLM decision call returns
        invalid JSON, before that decision fails safely.
    max_tool_top_k : int
        Server ceiling clamped onto `search_knowledge_base`'s `top_k`
        argument, independent of that argument's own Pydantic `le` bound.
        A config change, not just the schema literal, is always the
        true final ceiling.
    max_chunks_per_document_fetch : int
        Final chunk count `get_document`/`get_latest_document` keep after
        relevance-selecting against the current query.
    max_chunks_per_document_fetch_hard_ceiling : int
        SQL-level `LIMIT` applied to `get_document`/`get_latest_document`'s
        underlying fetch, before relevance selection. Bounds the read
        itself regardless of a document's actual size. Neither this nor
        `max_chunks_per_document_fetch` is exposed to the LLM in any tool
        schema.
    classify_prompt_path : str
        Path to the `classify_query` decision point's `PromptTemplate` YAML.
    decompose_prompt_path : str
        Path to the `decompose_query` decision point's `PromptTemplate` YAML.
    tool_select_prompt_path : str
        Path to the `select_tool` decision point's `PromptTemplate` YAML,
        used when `mcp.client.enabled=False` (offers only the 4 local
        tools).
    tool_select_mcp_prompt_path : str
        Path to the `select_tool` decision point's `PromptTemplate` YAML
        used instead when `mcp.client.enabled=True` (offers the 4 local
        tools plus `get_customer_case`/`get_case_status`). Kept as a
        separate template/path, not a runtime-conditional insertion into
        one template, so the 4-local-tool prompt an operator has already
        reviewed is byte-identical whether or not MCP client support
        happens to be compiled into this config.
    tool_select_mcp_actions_prompt_path : str
        Path to the `select_tool` decision point's `PromptTemplate` YAML
        used instead when `mcp.client.enabled=True` and `mcp.
        business_actions.enabled=True` (offers all 6 read tools plus
        `update_case_status`). A third, separate template file, not a
        further runtime-conditional insertion into
        `tool_select_mcp_prompt_path`.
    tool_select_graph_prompt_path : str
        Path to the `select_tool` decision point's `PromptTemplate` YAML
        used instead when `graph.enabled=True` and neither MCP variant
        applies (offers the 4 local RAG tools plus the 4 graph tools:
        `search_knowledge_graph`/`get_entity`/`get_neighbors`/
        `find_dependency_path`). See `rag.agent.graph._load_templates`
        for the full precedence order.
    evidence_sufficiency_prompt_path : str
        Path to the `evidence_sufficiency` decision point's `PromptTemplate` YAML.
    synthesize_prompt_path : str
        Path to the `synthesize` decision point's `PromptTemplate` YAML.
    """

    enabled: bool = False
    max_agent_steps: int = 8
    max_retrieval_attempts: int = 2
    max_tool_calls: int = 6
    max_json_parse_retries: int = 1
    max_tool_top_k: int = 20
    max_chunks_per_document_fetch: int = 10
    max_chunks_per_document_fetch_hard_ceiling: int = 50
    classify_prompt_path: str = "src/rag/prompts/templates/agent_classify_v1.yaml"
    decompose_prompt_path: str = "src/rag/prompts/templates/agent_decompose_v1.yaml"
    tool_select_prompt_path: str = "src/rag/prompts/templates/agent_tool_select_v2.yaml"
    tool_select_mcp_prompt_path: str = "src/rag/prompts/templates/agent_tool_select_v4_mcp.yaml"
    tool_select_mcp_actions_prompt_path: str = (
        "src/rag/prompts/templates/agent_tool_select_v5_mcp_actions.yaml"
    )
    tool_select_graph_prompt_path: str = "src/rag/prompts/templates/agent_tool_select_v6_graph.yaml"
    evidence_sufficiency_prompt_path: str = (
        "src/rag/prompts/templates/agent_evidence_sufficiency_v1.yaml"
    )
    synthesize_prompt_path: str = "src/rag/prompts/templates/agent_synthesize_v1.yaml"


class OllamaVisionConfig(BaseModel):
    """Model settings for the `ollama` vision provider.

    Reuses `generation.base_url_env_var`'s resolved URL
    (`AppConfig.ollama_base_url()`) rather than declaring a second env
    var: vision and text generation are the same native-host Ollama
    server, just different models. `model_name` defaults to `moondream`
    (~1.7GB), chosen for this project's CPU-only 8GB-RAM box over
    heavier options like `llava:7b`; the operator must `ollama pull` it
    themselves, same as `generation.model_name`.
    """

    model_name: str = "moondream"
    prompt_version: str = "v1"


class VisionConfig(BaseModel):
    """Vision provider selection for image-derived indexable content.

    `"none"` (the default) means no image bytes are ever read and no
    VisionProvider is constructed; text-only image handling, based on
    alt text and captions, works with this config untouched. `"ollama"`
    is the first real provider: a local, offline model call (no hosted
    egress, so `security.egress_policy` is not involved), matching this
    project's "production generation stays local-only" convention.
    """

    provider: Literal["none", "ollama"] = "none"
    ollama: OllamaVisionConfig = Field(default_factory=OllamaVisionConfig)


class McpServerConfig(BaseModel):
    """Streamable-HTTP mount settings for the MCP server.

    Attributes
    ----------
    mount_path : str
        Path the MCP ASGI app is mounted under, inside the existing
        `rag-api` FastAPI process (not a second server/process).
    """

    mount_path: str = "/mcp"


class McpClientConfig(BaseModel):
    """Agent-side MCP client settings for the two remote business tools.

    The four local RAG tools (`search_knowledge_base`/`get_document`/
    `get_latest_document`/`get_related_context`) never go through this;
    it exists only so `run_agent()` can dispatch `get_customer_case`/
    `get_case_status` as a real MCP client call, rather than a direct
    in-process function call.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a true no-op: the tool-select prompt
        never offers the two remote tools, `ToolSelectionDecision` never
        dispatches to them, and no MCP client/session is ever
        constructed. Matches `mcp.enabled`/`agent.enabled`'s convention.
    transport : {"asgi", "http"}
        `"asgi"` (the default) calls the same in-process MCP server
        object `main.py` mounts, via an in-memory ASGI transport: no
        real socket or port, still the full Streamable HTTP/JSON-RPC
        protocol. Requires `mcp.enabled=True` (there must be a server
        object to bind to); checked at startup, not request time.
        `"http"` is the escape hatch for a genuinely separate
        deployment: a real network call to `server_url`.
    server_url_env_var : str
        Environment variable naming the remote MCP server's base URL,
        read only when `transport="http"`. Mirrors
        `generation.base_url_env_var`'s `*_env_var` convention.
    default_server_url : str
        Fallback used when `server_url_env_var` is unset, for local/dev
        convenience (same-process loopback on the conventional API port).
    timeout_seconds : float
        Per-call MCP timeout (connection plus tool-call read timeout).
        A remote hang must never hang the whole agent run.
    internal_token_ttl_seconds : int
        Lifetime of the short-lived internal service token minted for
        each remote call when `security.auth.enabled=True` (see
        `rag.agent.mcp_client`). Short by design: minted immediately
        before use and consumed immediately.
    internal_token_subject : str
        The `sub` claim on a minted internal token: a fixed, clearly
        synthetic value, never the original caller's identity. Lets an
        audit-log reader distinguish an agent-minted service token from
        a real end-user token at a glance.
    internal_token_issuer : str
        Informational `iss` claim identifying this as an agent-to-MCP-
        service token, used only as a fallback when
        `security.auth.jwt.issuer` is unset (when it is set, the mint
        uses that configured value instead so `verify_jwt` actively
        validates it). There is no equivalent `internal_token_audience`
        fallback: PyJWT rejects any token carrying an `aud` claim when
        the caller passes no expected audience to check against, unlike
        `iss`, which is never inspected when
        `security.auth.jwt.audience` is unset. `aud` is included on a
        minted token only when `security.auth.jwt.audience` is actually
        configured; see `rag.agent.mcp_client.mint_internal_token`.
    """

    enabled: bool = False
    transport: Literal["asgi", "http"] = "asgi"
    server_url_env_var: str = "MCP_CLIENT_SERVER_URL"
    default_server_url: str = "http://127.0.0.1:8000/mcp"
    timeout_seconds: float = 10.0
    internal_token_ttl_seconds: int = 60
    internal_token_subject: str = "rag-agent-internal"
    internal_token_issuer: str = "rag-agent-internal"


class BusinessActionConfig(BaseModel):
    """Config for the `update_case_status` write-action MCP tool.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a true no-op: `update_case_status`
        is not registered on the MCP server at all, and a decision
        naming it fails closed in `_execute_tool` regardless. Matches
        `mcp.enabled`/`mcp.client.enabled`'s convention.
    approval_roles : list[str]
        Role names allowed to pre-authorize a sensitive case-status
        transition (e.g. `resolved -> closed`) via a request's
        `case_approvals` field. Independent of
        `security.authorization.cross_tenant_support_roles`: this gates
        an elevated write capability, not a tenant boundary crossing.
    max_case_approvals_per_request : int
        Operator-configurable ceiling on how many approvals a single
        request may carry, enforced at the API boundary (see
        `rag.api.request_auth.enforce_case_approval_limits`). Smaller
        than `rag.mcp.business.schemas.MAX_CASE_APPROVALS`, the hard
        schema-level ceiling that applies regardless of this value.
    """

    enabled: bool = False
    approval_roles: list[str] = Field(default_factory=lambda: ["case_status_approver"])
    max_case_approvals_per_request: int = 5


class McpConfig(BaseModel):
    """MCP server exposing the four RAG tools plus synthetic business-case tools.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), a true no-op: `main.py` never mounts
        the MCP ASGI app at all, so `mcp.server.mount_path` 404s like any
        other undefined route, and the `mcp` package's session-manager
        lifespan never starts. Matches `agent.enabled`/
        `security.authorization.enabled`'s convention.
    server : McpServerConfig
        Streamable-HTTP mount settings.
    client : McpClientConfig
        Agent-side MCP client settings for the remote business tools.
        Independent of `enabled`/`server` in principle (a genuinely
        external deployment could run the client against a server this
        process never mounts, via `transport="http"`), but the default
        `transport="asgi"` requires `enabled=True`.
    business_actions : BusinessActionConfig
        Settings for the `update_case_status` write-action tool. A
        separate kill-switch from `client.enabled`: the two read tools
        can be enabled without exposing the write tool.

    Notes
    -----
    Identity is governed entirely by `security.auth`; there is no
    separate `mcp.auth.*` block, so an MCP tool call is authenticated by
    the same `security.auth.enabled`/`insecure_dev_mode`/`jwt` rules
    `POST /query` already uses (see `rag.mcp.identity`).

    `get_customer_case`/`get_case_status` (`rag.mcp.business.store`) have
    no config toggle of their own and their tenant/role authorization is
    always enforced, with no equivalent of
    `security.authorization.enabled`'s kill-switch. Because that
    authorization depends on a trustworthy identity, `mcp.client.enabled=True`
    requires `security.auth.enabled=True`, checked at startup (see
    `rag.agent.mcp_client.validate_startup_config`).
    """

    enabled: bool = False
    server: McpServerConfig = Field(default_factory=McpServerConfig)
    client: McpClientConfig = Field(default_factory=McpClientConfig)
    business_actions: BusinessActionConfig = Field(default_factory=BusinessActionConfig)


class FeedbackConfig(BaseModel):
    """User-feedback-loop persistence tunables (`POST /feedback`).

    Attributes
    ----------
    enabled : bool
        When `False`, `POST /feedback` 404s, matching the router-level
        runtime-check pattern `security.dos_limits`/`observability.
        live_events` already use. `True` by default: feedback collection
        has no external dependency beyond the already-required Postgres
        instance.
    table_name : str
        Name of the feedback table.
    store_query_text : bool
        Whether to persist the caller-echoed original query text. The
        caller already saw this text in their own request; storing it is
        what makes a feedback row usable for later eval-curation review.
        An operator with a stricter privacy posture can turn this off.
    store_answer_text : bool
        Same reasoning as `store_query_text`, for the generated answer
        text the caller already received.
    max_comment_length : int
        Maximum allowed free-text `comment` length, in characters.
    max_query_text_length : int
        Maximum allowed `query` length, in characters. Mirrors
        `security.dos_limits.max_query_length` but kept independent since
        feedback is a different endpoint with its own bound.
    max_answer_text_length : int
        Maximum allowed `answer` length, in characters.
    minconn, maxconn : int
        `psycopg2.pool.ThreadedConnectionPool` bounds for `FeedbackStore`'s
        own, deliberately small and separate pool (see that class's
        docstring). Matches the previous hardcoded defaults (`minconn=1`,
        `maxconn=3`).
    max_cited_sources : int
        Maximum number of entries allowed in `cited_source_ids`/
        `tool_calls`.

    Notes
    -----
    Enforced as router-level checks (`api/routers/feedback.py`'s
    `_enforce_feedback_limits`), not Pydantic field constraints, matching
    `DoSLimitsConfig`'s own reasoning: the bounds must read from runtime
    config, not a value baked into the request model at class-definition
    time.
    """

    enabled: bool = True
    table_name: str = "feedback"
    store_query_text: bool = True
    store_answer_text: bool = True
    max_comment_length: int = 1000
    max_query_text_length: int = 2000
    max_answer_text_length: int = 8000
    max_cited_sources: int = 20
    minconn: int = 1
    maxconn: int = 3


class Neo4jConnectionConfig(BaseModel):
    """Neo4j driver connection and pooling settings.

    Attributes
    ----------
    uri_env_var, user_env_var, password_env_var : str
        Names of the environment variables that hold the connection
        settings. The values themselves are never stored in config.
    database : str
        Database name within the DBMS.
    max_connection_pool_size : int
        Upper bound on the driver's connection pool.
    connection_acquisition_timeout_seconds : float
        How long a query waits for a pooled connection before failing.
    connection_timeout_seconds : float
        How long a new connection handshake may take.
    max_transaction_retry_time_seconds : float
        How long the driver retries transient errors. Client errors, such
        as a syntax error, are not retried.
    """

    uri_env_var: str = "NEO4J_URI"
    user_env_var: str = "NEO4J_USER"
    password_env_var: str = "NEO4J_PASSWORD"
    database: str = "neo4j"
    max_connection_pool_size: int = 20
    connection_acquisition_timeout_seconds: float = 10.0
    connection_timeout_seconds: float = 5.0
    max_transaction_retry_time_seconds: float = 15.0


class GraphTraversalConfig(BaseModel):
    """Bounds applied to graph queries.

    Every query uses an explicit upper bound on traversal depth. Unbounded
    variable-length patterns are never issued.

    Attributes
    ----------
    max_traversal_depth : int
        Upper bound on relationship hops.
    max_nodes_returned : int
        Maximum rows returned by one neighbor or entity query.
    max_paths_returned : int
        Declared but not enforced. The shortest-path query returns at most one path.
    query_timeout_seconds : float
        Server-side transaction timeout for each query.
    """

    max_traversal_depth: int = 3
    max_nodes_returned: int = 50
    max_paths_returned: int = 10
    query_timeout_seconds: float = 5.0


class GraphExtractionConfig(BaseModel):
    """Settings for graph extraction during ingestion.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), ingestion writes no graph facts. Graph
        retrieval can still read an existing graph.
    llm_extraction_enabled : bool
        When `False`, only deterministic facts are written: `Document`
        nodes, `PART_OF`, and `SUPERSEDES`. When `True`, one LLM call per
        chunk also extracts entities and relationships.
    min_confidence : float
        LLM relationships below this confidence are dropped before writing.
    max_relationships_per_chunk : int
        Maximum relationships kept from one LLM call, applied after validation.
    extraction_prompt_path : str
        Path to the extraction prompt template.
    """

    enabled: bool = False
    llm_extraction_enabled: bool = True
    min_confidence: float = 0.5
    max_relationships_per_chunk: int = 8
    extraction_prompt_path: str = "src/rag/prompts/templates/graph_extract_v1.yaml"


class CypherGenerationConfig(BaseModel):
    """Settings for LLM-generated Cypher, used when no deterministic template matches.

    Attributes
    ----------
    enabled : bool
        When `False` (the default), only the built-in query templates run.
        When `True`, the LLM may write read-only Cypher, which is validated
        by `rag.graph.cypher_guard` before it runs.
    max_result_limit : int
        Every generated query gets a `LIMIT` with this value.
    max_depth : int
        Maximum upper bound allowed on a variable-length pattern.
    timeout_seconds : float
        Transaction timeout for generated queries. Separate from
        `GraphTraversalConfig.query_timeout_seconds`.
    prompt_path : str
        Path to the Cypher-generation prompt template.
    """

    enabled: bool = False
    max_result_limit: int = 50
    max_depth: int = 3
    timeout_seconds: float = 5.0
    prompt_path: str = "src/rag/prompts/templates/graph_cypher_generate_v1.yaml"


class GraphRAGFusionConfig(BaseModel):
    """How graph evidence is combined with vector/hybrid evidence in `answer()`.

    Attributes
    ----------
    strategy : {"vector_only", "graph_only", "vector_plus_graph", "graph_seeded_vector"}
        `"vector_only"` (the default) returns the normal vector answer.
        `"graph_only"` uses graph facts alone. `"vector_plus_graph"` runs both
        and appends graph facts after the vector results. `"graph_seeded_vector"`
        boosts vector results from documents linked to query entities. It
        reorders results and never removes any.
    max_graph_evidence_items : int
        Maximum graph facts added to one answer's context.
    graph_seed_top_k : int
        Number of linked entities used to seed `"graph_seeded_vector"`.
    """

    strategy: Literal["vector_only", "graph_only", "vector_plus_graph", "graph_seeded_vector"] = (
        "vector_only"
    )
    max_graph_evidence_items: int = 5
    graph_seed_top_k: int = 3


class GraphConfig(BaseModel):
    """Root config block for the optional Knowledge Graph / GraphRAG capability.

    Attributes
    ----------
    enabled : bool
        Master switch. When `False` (the default), no Neo4j driver is
        constructed, ingestion writes no graph facts, and the graph agent
        tools are not offered to the model.
    neo4j : Neo4jConnectionConfig
        Connection and pooling settings.
    traversal : GraphTraversalConfig
        Bounds applied to graph queries.
    extraction : GraphExtractionConfig
        Settings for graph extraction during ingestion.
    cypher_generation : CypherGenerationConfig
        Settings for LLM-generated Cypher.
    fusion : GraphRAGFusionConfig
        How graph evidence is combined with vector evidence.
    """

    enabled: bool = False
    neo4j: Neo4jConnectionConfig = Field(default_factory=Neo4jConnectionConfig)
    traversal: GraphTraversalConfig = Field(default_factory=GraphTraversalConfig)
    extraction: GraphExtractionConfig = Field(default_factory=GraphExtractionConfig)
    cypher_generation: CypherGenerationConfig = Field(default_factory=CypherGenerationConfig)
    fusion: GraphRAGFusionConfig = Field(default_factory=GraphRAGFusionConfig)


class TokenCountingConfig(BaseModel):
    """Token-estimation strategy, used before any text reaches the LLM.

    See `rag.context.tokens`'s module docstring for why this is an
    estimate, never an exact provider count, and why that is
    unavoidable for a local Ollama deployment.

    Attributes
    ----------
    strategy : {"heuristic", "tiktoken"}
        `"heuristic"` (the default) needs no extra dependency.
        `"tiktoken"` falls back to `"heuristic"` with a logged warning
        if the optional `tiktoken` package is not installed.
    chars_per_token : float
        Used by the heuristic strategy.
    tiktoken_encoding : str
        Used by the tiktoken strategy.
    """

    strategy: Literal["heuristic", "tiktoken"] = "heuristic"
    chars_per_token: float = 4.0
    tiktoken_encoding: str = "cl100k_base"


class ModelContextLimitsConfig(BaseModel):
    """Per-model context-window table, keyed by `generation.model_name`.

    Attributes
    ----------
    default_context_window_tokens : int
        Used for any `generation.model_name` with no entry in
        `overrides`. 32768 matches the qwen2.5 family's trained context
        length; a deployment running a different model should add an
        explicit override rather than rely on this default being correct.
    overrides : dict[str, int]
        `{model_name: context_window_tokens}`.
    """

    default_context_window_tokens: int = 32768
    overrides: dict[str, int] = Field(
        default_factory=lambda: {
            "qwen2.5:0.5b": 32768,
            "qwen2.5:1.5b": 32768,
            "qwen2.5:3b": 32768,
            "qwen2.5:7b": 32768,
            "llama3.1:8b": 131072,
            "moondream": 2048,
        }
    )


class SourceBudgetConfig(BaseModel):
    """Fractional share of `available_context_tokens` given to each evidence bucket.

    Shares must sum to 1.0 (validated). `tool_fetched` evidence (an
    agent's `get_document`/`get_latest_document`/`get_related_context`
    direct fetches) shares the `document` bucket, since it is drawn from
    the same corpus as retrieved chunks, not a fourth bucket.

    Attributes
    ----------
    document_share : float
        Vector/hybrid retrieval evidence, plus tool-fetched document
        chunks.
    graph_path_share : float
        GraphRAG evidence (see `rag.context.graph_packing`).
    mcp_tool_share : float
        MCP/business-tool evidence.
    conversation_history_share : float
        Prior conversation turns (see `rag.context.history`).
    max_document_tokens, max_graph_tokens, max_tool_tokens, max_history_tokens : int | None
        Optional hard per-bucket ceilings, applied after the
        share-based allocation. `None` (the default) means "no
        additional ceiling beyond the share itself". A defense-in-depth
        knob, since the shares already sum to 1.0 and so cannot let one
        source consume the whole budget on their own; a ceiling is
        useful mainly when `available_context_tokens` is unusually
        large and a hard cap is still wanted for one bucket.
    """

    document_share: float = 0.55
    graph_path_share: float = 0.15
    mcp_tool_share: float = 0.15
    conversation_history_share: float = 0.15
    max_document_tokens: int | None = None
    max_graph_tokens: int | None = None
    max_tool_tokens: int | None = None
    max_history_tokens: int | None = None

    def shares_sum_to_one(self) -> bool:
        """Return whether the four shares sum to 1.0 within floating-point tolerance."""
        total = (
            self.document_share
            + self.graph_path_share
            + self.mcp_tool_share
            + self.conversation_history_share
        )
        return abs(total - 1.0) < 1e-6


class AdaptiveBudgetConfig(BaseModel):
    """Query-type-dependent `SourceBudgetConfig` overrides.

    Attributes
    ----------
    enabled : bool
        When `False`, `source_budgets` (the fixed allocation) is always
        used regardless of query composition. When `True` (the
        default), `rag.context.planner` classifies each request into
        one of `overrides`' keys (`"knowledge_only"`, `"relationship"`,
        `"business"`, `"conversational"`, or falls back to the fixed
        `source_budgets` for `"general"`) from the evidence actually
        gathered, never from a second LLM call.
    overrides : dict[str, SourceBudgetConfig]
        Each value's shares must independently sum to 1.0.
    """

    enabled: bool = True
    overrides: dict[str, SourceBudgetConfig] = Field(
        default_factory=lambda: {
            "knowledge_only": SourceBudgetConfig(
                document_share=0.75,
                graph_path_share=0.05,
                mcp_tool_share=0.05,
                conversation_history_share=0.15,
            ),
            "relationship": SourceBudgetConfig(
                document_share=0.35,
                graph_path_share=0.45,
                mcp_tool_share=0.05,
                conversation_history_share=0.15,
            ),
            "business": SourceBudgetConfig(
                document_share=0.25,
                graph_path_share=0.05,
                mcp_tool_share=0.55,
                conversation_history_share=0.15,
            ),
            "conversational": SourceBudgetConfig(
                document_share=0.35,
                graph_path_share=0.05,
                mcp_tool_share=0.05,
                conversation_history_share=0.55,
            ),
        }
    )


class DeduplicationConfig(BaseModel):
    """Context deduplication tunables. See `rag.context.dedup`.

    Attributes
    ----------
    enabled : bool
    overlap_threshold : float
        Normalized-text Jaccard-overlap ratio (over whitespace-
        tokenized word shingles) above which two items are treated as
        near-duplicates, in `[0.0, 1.0]`. Exact `chunk_id` matches and
        identical content hashes are always deduplicated regardless of
        this threshold; it only governs the fuzzier "high overlap, not
        identical" case (e.g. the same document fact surfaced via
        direct retrieval and again via relationship expansion or a
        graph fact).
    """

    enabled: bool = True
    overlap_threshold: float = 0.85


class CompressionConfig(BaseModel):
    """Oversized-evidence compression tunables. See `rag.context.compression`.

    Attributes
    ----------
    enabled : bool
    threshold_tokens : int
        A floor on how small a *remaining headroom* must be for
        compression to even be attempted, not on how large the
        candidate item itself is. An item that doesn't fit its
        headroom is always a compression candidate, regardless of the
        item's own size, as long as that headroom is at least this
        many tokens; compressing something down to a 2-token sliver
        would not produce usable evidence.
    strategy : {"sentence_trim", "extractive"}
        `"sentence_trim"` keeps leading/trailing sentences up to the
        target budget. `"extractive"` (the default) additionally scores
        sentences by query-term overlap and keeps the highest-scoring
        ones in their original order, a deterministic, dependency-free
        stand-in for "select relevant sections" (see that module's
        docstring for why a second LLM call was not used here).
    max_sentences : int
        Ceiling on sentences kept by either strategy, independent of
        the token budget (keeps a compressed item from being an
        unreadable wall of comma-free sentence fragments even when the
        budget would technically allow more).
    """

    enabled: bool = True
    threshold_tokens: int = 30
    strategy: Literal["sentence_trim", "extractive"] = "extractive"
    max_sentences: int = 8


class HistoryConfig(BaseModel):
    """Conversation-history management tunables. See `rag.context.history`.

    Attributes
    ----------
    strategy : {"full", "sliding_window", "summarized", "retrieval"}
        `"full"`: include every supplied turn (bounded only by the
        `conversation_history` budget bucket itself). `"sliding_window"`
        (the default): keep only the most recent `max_turns`.
        `"summarized"`: summarize everything older than `max_turns` into
        one compact item, keep recent turns raw. `"retrieval"`:
        keyword-overlap-select the most query-relevant older turns
        instead of a simple recency window, keep recent turns raw
        either way.
    max_turns : int
        Recent turns kept raw under every strategy except `"full"`.
    summarize_min_turns_to_trigger : int
        `"summarized"` only summarizes once there are more than this
        many older turns; below it, behaves like `"sliding_window"`.
    """

    strategy: Literal["full", "sliding_window", "summarized", "retrieval"] = "sliding_window"
    max_turns: int = 6
    summarize_min_turns_to_trigger: int = 4


class ContextCacheConfig(BaseModel):
    """In-process, request-scoped safe caching. See `rag.context.caching`.

    Attributes
    ----------
    enabled : bool
        Default `False`: no measurable benefit was demonstrated for this
        single-instance, CPU-bound deployment (see
        `docs/context_engineering.md`'s caching section), so this stays
        off rather than adding a dependency speculatively, matching this
        project's established "no infrastructure without a demonstrated
        requirement now" convention. When enabled, caches only
        per-text token counts (never raw evidence, never a full
        assembled context) in a small in-process LRU, keyed by a hash of
        `(tenant_id, text)` so no cache entry can ever cross a tenant
        boundary.
    max_entries : int
        LRU size cap.
    """

    enabled: bool = False
    max_entries: int = 2048


class ContextEngineeringConfig(BaseModel):
    """Root config block for the Context Engineering subsystem (`rag.context`).

    Attributes
    ----------
    enabled : bool
        When `False` (the default), every call site
        (`RetrievalPipeline.answer`, `GraphRAGPipeline`, the agent's
        `_synthesize` node) behaves exactly as it did before this
        subsystem existed: `rag.retrieval.pipeline.build_context` is
        called directly, with no budgeting, deduplication, or
        compression at all. A true no-op, matching every other
        milestone's kill-switch convention in this codebase.
    reserved_output_tokens : int | None
        Overrides `rag.context.model_limits.ModelContextLimits.
        default_reserved_output_tokens` (which otherwise mirrors
        `generation.max_tokens`) when set.
    framework_overhead_tokens : int
        A fixed allowance for chat-template/role-formatting overhead
        Ollama's `chat()` endpoint adds per message, which a raw-text
        token count does not capture.
    safety_margin_tokens : int
        Extra buffer absorbing token-estimation error (see
        `rag.context.tokens`).
    token_counting : TokenCountingConfig
    model_limits : ModelContextLimitsConfig
    source_budgets : SourceBudgetConfig
        The fixed allocation, used when `adaptive.enabled=False` or as
        the `"general"` fallback when adaptive allocation is on but no
        bucket type dominates the gathered evidence.
    adaptive : AdaptiveBudgetConfig
    deduplication : DeduplicationConfig
    compression : CompressionConfig
    history : HistoryConfig
    packing_strategy : {"top_n", "naive_token_first", "priority_aware"}
        `"top_n"` reproduces the pre-existing fixed-chunk-count
        behavior (for the benchmark comparison in
        `docs/context_engineering.md`, not meant for production use
        once this is enabled). `"naive_token_first"` packs by rank order
        until the budget is exhausted, ignoring authority/freshness.
        `"priority_aware"` (the default when enabled) is the full
        algorithm: see `rag.context.packing`.
    ordering_strategy : {"priority_first", "authority_first", "begin_end_sandwich",
            "grouped_by_source"}
        How packed items are ordered in the final rendered text; see
        `rag.context.ordering`. Independent of `packing_strategy` (which
        decides *what* survives the budget): `"begin_end_sandwich"` (the
        default) places the highest-priority evidence at both the start
        and end of the context block, per the "lost in the middle"
        experiment in `docs/context_engineering.md`.
    cache : ContextCacheConfig
    """

    enabled: bool = False
    reserved_output_tokens: int | None = None
    framework_overhead_tokens: int = 128
    safety_margin_tokens: int = 256
    token_counting: TokenCountingConfig = Field(default_factory=TokenCountingConfig)
    model_limits: ModelContextLimitsConfig = Field(default_factory=ModelContextLimitsConfig)
    source_budgets: SourceBudgetConfig = Field(default_factory=SourceBudgetConfig)
    adaptive: AdaptiveBudgetConfig = Field(default_factory=AdaptiveBudgetConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    packing_strategy: Literal["top_n", "naive_token_first", "priority_aware"] = "priority_aware"
    ordering_strategy: Literal[
        "priority_first", "authority_first", "begin_end_sandwich", "grouped_by_source"
    ] = "begin_end_sandwich"
    cache: ContextCacheConfig = Field(default_factory=ContextCacheConfig)

    @model_validator(mode="after")
    def _budget_shares_sum_to_one(self) -> ContextEngineeringConfig:
        """Fail fast if `source_budgets` or any `adaptive.overrides` entry doesn't sum to 1.0."""
        if not self.source_budgets.shares_sum_to_one():
            raise ValueError("context.source_budgets' four shares must sum to 1.0")
        for name, budget in self.adaptive.overrides.items():
            if not budget.shares_sum_to_one():
                raise ValueError(f"context.adaptive.overrides[{name!r}]'s shares must sum to 1.0")
        return self


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
    agent: AgentConfig = Field(default_factory=AgentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    context: ContextEngineeringConfig = Field(default_factory=ContextEngineeringConfig)

    def agent_prompt_template_path(self, path: str) -> Path:
        """Resolve one of `agent.*_prompt_path`, relative to the repo root if not absolute.

        Parameters
        ----------
        path : str
            One of `agent.classify_prompt_path`/`decompose_prompt_path`/
            `tool_select_prompt_path`/`evidence_sufficiency_prompt_path`/
            `synthesize_prompt_path`.

        Returns
        -------
        Path
            The resolved path to the prompt template YAML file.
        """
        candidate = Path(path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

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

    def neo4j_uri(self) -> str:
        """Resolve the Neo4j Bolt URI from `graph.neo4j.uri_env_var`.

        Returns
        -------
        str
            The URI read from the configured environment variable, or
            ``"bolt://localhost:7687"`` (the native/host default,
            matching `docker-compose.graph.yml`'s published port) if
            unset.
        """
        return os.environ.get(self.graph.neo4j.uri_env_var, "bolt://localhost:7687")

    def neo4j_user(self) -> str:
        """Resolve the Neo4j username from `graph.neo4j.user_env_var`.

        Returns
        -------
        str
            The username read from the configured environment variable,
            or ``"neo4j"`` (the image's own default account name) if
            unset.
        """
        return os.environ.get(self.graph.neo4j.user_env_var, "neo4j")

    def neo4j_password(self) -> str:
        """Resolve the Neo4j password from `graph.neo4j.password_env_var`.

        Returns
        -------
        str
            The password read from the environment.

        Raises
        ------
        RuntimeError
            If that environment variable is unset or empty. Unlike
            `neo4j_user`, there is no silent default: a graph client
            must never be constructed with an empty credential.
        """
        value = os.environ.get(self.graph.neo4j.password_env_var)
        if not value:
            raise RuntimeError(
                f"Environment variable '{self.graph.neo4j.password_env_var}' is not set. "
                "Set NEO4J_PASSWORD to the Neo4j account password."
            )
        return value

    def mcp_client_server_url(self) -> str:
        """Resolve the remote MCP server base URL for `mcp.client.transport="http"`.

        Returns
        -------
        str
            The URL read from `mcp.client.server_url_env_var`, or
            `mcp.client.default_server_url` if that variable is unset.
            Unused in the default `transport="asgi"` mode, which binds
            directly to the in-process MCP ASGI app instead.
        """
        return os.environ.get(
            self.mcp.client.server_url_env_var, self.mcp.client.default_server_url
        )

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

    def otlp_endpoint(self) -> str:
        """Resolve the OTLP exporter endpoint for OpenTelemetry tracing.

        Returns
        -------
        str
            The URL read from `OTEL_EXPORTER_OTLP_ENDPOINT`, or
            ``"http://localhost:4318"`` (a native/uvicorn-on-host Jaeger)
            if unset. `docker-compose.observability.yml` overrides this
            to ``http://jaeger:4318`` for the containerized `rag-api`.
        """
        return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

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

    Raises
    ------
    RuntimeError
        If `path` does not exist or its contents are not valid YAML. Never
        falls back to `DEFAULT_CONFIG_PATH` on failure.
    """
    load_dotenv(override=False)
    resolved = Path(path)
    if not resolved.is_file():
        raise RuntimeError(f"Config file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Config file is not valid YAML: {resolved} ({exc})") from exc
    return AppConfig.model_validate(raw)
