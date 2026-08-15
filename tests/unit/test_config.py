from __future__ import annotations

import pytest

from rag.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    ChunkingConfig,
    HybridRetrievalConfig,
    load_config,
)


def test_load_default_config_has_expected_defaults():
    """config/default.yaml loads with its documented default values."""
    config = load_config()
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.generation.model_name == "qwen2.5:1.5b"
    assert config.vectorstore.connection_env_var == "DATABASE_URL"
    assert config.reranker.provider == "none"


def test_default_config_chunking_provider_is_structured_markdown():
    """config/default.yaml's chunking.provider defaults to structured_markdown with its tunables."""
    config = load_config()
    assert config.chunking.provider == "structured_markdown"
    assert config.chunking.structured_markdown.table_row_group_size == 20
    assert config.chunking.structured_markdown.max_atomic_block_chars == 2000


def test_chunking_config_structured_markdown_validates_with_defaults():
    """ChunkingConfig(provider='structured_markdown') validates with correct nested defaults."""
    config = ChunkingConfig(provider="structured_markdown")
    assert config.structured_markdown.table_row_group_size == 20
    assert config.structured_markdown.max_atomic_block_chars == 2000


def test_database_url_resolves_from_env(monkeypatch):
    """database_url() reads DATABASE_URL from the process environment."""
    config = load_config()
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:15987/ragdb")
    assert config.database_url() == "postgresql://rag:rag@localhost:15987/ragdb"


def test_database_url_raises_clear_error_when_unset(monkeypatch):
    """database_url() raises a RuntimeError naming the env var when unset."""
    config = load_config()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        config.database_url()


def test_ollama_base_url_has_sane_default(monkeypatch):
    """ollama_base_url() falls back to localhost when its env var is unset."""
    config = load_config()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert config.ollama_base_url() == "http://localhost:11434"


def test_judge_default_provider_is_not_a_generation_model():
    """The default judge provider/model never matches the generation model."""
    config = load_config()
    assert config.judge.provider == "openai"
    assert config.judge.ollama.model_name not in {"qwen2.5:1.5b", "qwen2.5:3b"}
    assert config.judge.ollama.model_name != config.generation.model_name


def test_judge_cache_enabled_by_default_under_dot_cache_ragas():
    """Judge-call caching defaults on, storing under the gitignored .cache/ tree."""
    config = load_config()
    assert config.judge.cache_enabled is True
    assert config.judge.cache_dir == ".cache/ragas"


def test_openai_api_key_resolves_from_env(monkeypatch):
    """openai_api_key() reads OPENAI_API_KEY from the process environment."""
    config = load_config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    assert config.openai_api_key() == "sk-test-123"


def test_anthropic_api_key_resolves_from_env(monkeypatch):
    """anthropic_api_key() reads ANTHROPIC_API_KEY from the process environment."""
    config = load_config()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    assert config.anthropic_api_key() == "sk-ant-test-123"


def test_default_config_mlflow_is_enabled_with_local_sqlite_backend():
    """config/default.yaml enables MLflow by default, using a local SQLite-backed tracking store."""
    config = load_config()
    assert config.mlflow.enabled is True
    assert config.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert config.mlflow.experiment_name == "local-rag-system"


def test_default_config_retrieval_provider_is_dense():
    """config/default.yaml's retrieval.provider defaults to dense-only, hybrid tunables ready."""
    config = load_config()
    assert config.retrieval.provider == "dense"
    assert config.retrieval.hybrid.rrf_k == 60


def test_hybrid_retrieval_config_defaults():
    """HybridRetrievalConfig() has the standard RRF k=60 default."""
    assert HybridRetrievalConfig().rrf_k == 60


def test_retrieval_config_hybrid_validates_with_custom_rrf_k():
    """RetrievalConfig(provider='hybrid') accepts a custom nested rrf_k."""
    config = load_config().model_copy(deep=True)
    config.retrieval.provider = "hybrid"
    config.retrieval.hybrid.rrf_k = 30
    assert config.retrieval.provider == "hybrid"
    assert config.retrieval.hybrid.rrf_k == 30


def test_default_config_relationship_expansion_disabled_by_default():
    """config/default.yaml's relationship_expansion is off, a no-op unless explicitly enabled."""
    config = load_config()
    expansion = config.retrieval.relationship_expansion
    assert expansion.enabled is False
    assert expansion.include_parent is True
    assert expansion.include_neighbors is True
    assert expansion.max_related_elements == 3


def test_default_config_vision_provider_is_none():
    """config/default.yaml's vision.provider defaults to 'none' -- no image bytes ever read."""
    config = load_config()
    assert config.vision.provider == "none"


def test_multimodal_v2_text_only_experiment_config_loads_with_expected_overrides():
    """The Experiment A config activates prompt v2 + hybrid retrieval, expansion off."""
    config = load_config("config/experiments/multimodal-v2-text-only.yaml")
    assert config.generation.prompt.version == "v2"
    assert config.retrieval.provider == "hybrid"
    assert config.reranker.provider == "none"
    assert config.retrieval.relationship_expansion.enabled is False
    assert config.vision.provider == "none"
    assert config.generation.model_name == "qwen2.5:1.5b"


def test_multimodal_v2_relationship_experiment_config_only_differs_by_expansion():
    """The Experiment B config matches Experiment A except relationship_expansion.enabled=True."""
    config_a = load_config("config/experiments/multimodal-v2-text-only.yaml")
    config_b = load_config("config/experiments/multimodal-v2-relationship.yaml")
    assert config_b.retrieval.relationship_expansion.enabled is True
    a_dict = config_a.model_dump(exclude={"retrieval": {"relationship_expansion": {"enabled"}}})
    b_dict = config_b.model_dump(exclude={"retrieval": {"relationship_expansion": {"enabled"}}})
    assert a_dict == b_dict


def test_stage_b_cross_encoder_config_only_differs_by_reranker_from_baseline():
    """The Stage B candidate config isolates the reranker: everything else matches the baseline.

    multimodal-v2-relationship-qwen3b.yaml is the Stage B baseline (it's
    experiment_015's own config, unchanged in value, just migrated to the
    new candidate_k/generation_context_top_n field names). The candidate
    config must change reranker.provider/reranker.top_n only.
    """
    baseline = load_config("config/experiments/multimodal-v2-relationship-qwen3b.yaml")
    candidate = load_config(
        "config/experiments/multimodal-v2-relationship-qwen3b_cross-encoder.yaml"
    )

    assert baseline.reranker.provider == "none"
    assert candidate.reranker.provider == "cross_encoder"
    assert candidate.reranker.top_n == 5
    assert baseline.retrieval.candidate_k == candidate.retrieval.candidate_k == 5
    assert (
        baseline.retrieval.generation_context_top_n
        == candidate.retrieval.generation_context_top_n
        == 3
    )

    baseline_dict = baseline.model_dump(exclude={"reranker": {"provider", "top_n"}})
    candidate_dict = candidate.model_dump(exclude={"reranker": {"provider", "top_n"}})
    assert baseline_dict == candidate_dict


def test_generation_context_latency_configs_isolate_one_variable_at_a_time():
    """Config A -> B -> C (context/token-budget milestone) each change exactly one field.

    A = multimodal-v2-relationship-qwen3b.yaml (the established best
    retrieval/generation config). B changes only
    retrieval.generation_context_top_n (3 -> 2). C changes only
    generation.max_tokens on top of B (512 -> 256) -- everything else
    (embeddings, chunking, retrieval strategy, prompt, model) stays fixed
    across all three, per this milestone's controlled-experiment design.
    """
    config_a = load_config("config/experiments/multimodal-v2-relationship-qwen3b.yaml")
    config_b = load_config("config/experiments/multimodal-v2-relationship-qwen3b_ctx2.yaml")
    config_c = load_config("config/experiments/multimodal-v2-relationship-qwen3b_ctx2_tok256.yaml")

    assert config_a.retrieval.generation_context_top_n == 3
    assert config_b.retrieval.generation_context_top_n == 2
    assert config_c.retrieval.generation_context_top_n == 2
    assert config_a.generation.max_tokens == 512
    assert config_b.generation.max_tokens == 512
    assert config_c.generation.max_tokens == 256

    a_dict = config_a.model_dump(exclude={"retrieval": {"generation_context_top_n"}})
    b_dict = config_b.model_dump(exclude={"retrieval": {"generation_context_top_n"}})
    assert a_dict == b_dict

    b_dict_2 = config_b.model_dump(exclude={"generation": {"max_tokens"}})
    c_dict = config_c.model_dump(exclude={"generation": {"max_tokens"}})
    assert b_dict_2 == c_dict


def test_generation_config_seed_defaults_to_none():
    """GenerationConfig.seed is None (non-deterministic) unless a config file sets it."""
    config = load_config()
    assert config.generation.seed is None


def test_classic_rag_baseline_v1_changes_only_temperature_and_seed():
    """classic-rag-baseline-v1.yaml is experiment_015's config, made deterministic.

    Same retrieval/chunking/embedding/reranking/prompt-v2/relationship-
    expansion/model as multimodal-v2-relationship-qwen3b.yaml -- only
    generation.temperature (0.2 -> 0.0) and generation.seed (None -> 42)
    differ, per the classic-RAG-baseline milestone's explicit
    "don't change any other experiment variable" requirement.
    """
    baseline = load_config("config/experiments/multimodal-v2-relationship-qwen3b.yaml")
    classic = load_config("config/experiments/classic-rag-baseline-v1.yaml")

    assert baseline.generation.temperature == 0.2
    assert classic.generation.temperature == 0.0
    assert baseline.generation.seed is None
    assert classic.generation.seed == 42

    baseline_dict = baseline.model_dump(exclude={"generation": {"temperature", "seed"}})
    classic_dict = classic.model_dump(exclude={"generation": {"temperature", "seed"}})
    assert baseline_dict == classic_dict


def test_default_config_authorization_disabled_by_default():
    """config/default.yaml's security.authorization is off -- a no-op unless explicitly enabled."""
    config = load_config()
    assert config.security.authorization.enabled is False
    assert config.security.authorization.cross_tenant_support_roles == ["techfusion_support"]


def test_default_config_field_redaction_disabled_by_default():
    """config/default.yaml's security.field_redaction is off -- a no-op unless enabled."""
    config = load_config()
    assert config.security.field_redaction.enabled is False


def test_secure_rag_baseline_v1_field_redaction_changes_only_that_flag():
    """The field-redaction candidate isolates exactly one variable from experiment_026's config.

    Per the field-level-safety milestone's design review (adjustment 2):
    prompt stays v3, matching secure-rag-baseline-v1.yaml exactly, so
    field redaction is the only meaningful changed safety mechanism in
    this A/B.
    """
    control = load_config("config/experiments/secure-rag-baseline-v1.yaml")
    candidate = load_config("config/experiments/secure-rag-baseline-v1-field-redaction.yaml")

    assert control.generation.prompt.version == "v3"
    assert candidate.generation.prompt.version == "v3"
    assert control.security.field_redaction.enabled is False
    assert candidate.security.field_redaction.enabled is True

    control_dict = control.model_dump(exclude={"security": {"field_redaction": {"enabled"}}})
    candidate_dict = candidate.model_dump(exclude={"security": {"field_redaction": {"enabled"}}})
    assert control_dict == candidate_dict


def test_secure_rag_baseline_v2_jwt_auth_changes_only_the_auth_toggle():
    """The jwt-auth candidate isolates exactly the security.auth block from experiment_027's config.

    Auth-boundary milestone (approved adjustment 1): prompt stays v3,
    matching secure-rag-baseline-v1-field-redaction.yaml exactly, so the
    JWT-boundary change isn't confounded with a prompt-wording change.
    """
    control = load_config("config/experiments/secure-rag-baseline-v1-field-redaction.yaml")
    candidate = load_config("config/experiments/secure-rag-baseline-v2-jwt-auth.yaml")

    assert control.generation.prompt.version == "v3"
    assert candidate.generation.prompt.version == "v3"
    assert control.security.auth.enabled is False
    assert candidate.security.auth.enabled is True

    control_dict = control.model_dump(exclude={"security": {"auth"}})
    candidate_dict = candidate.model_dump(exclude={"security": {"auth"}})
    assert control_dict == candidate_dict


def test_secure_rag_baseline_v2_jwt_auth_dev_only_differs_by_insecure_dev_mode():
    """The -dev overlay differs from the primary jwt-auth candidate only by insecure_dev_mode."""
    primary = load_config("config/experiments/secure-rag-baseline-v2-jwt-auth.yaml")
    dev = load_config("config/experiments/secure-rag-baseline-v2-jwt-auth-dev.yaml")

    assert primary.security.auth.insecure_dev_mode is False
    assert dev.security.auth.insecure_dev_mode is True

    primary_dict = primary.model_dump(exclude={"security": {"auth": {"insecure_dev_mode"}}})
    dev_dict = dev.model_dump(exclude={"security": {"auth": {"insecure_dev_mode"}}})
    assert primary_dict == dev_dict


def test_secure_rag_baseline_v1_changes_only_prompt_and_authorization():
    """secure-rag-baseline-v1.yaml is classic-rag-baseline-v1.yaml + prompt v3 + auth enabled.

    No other variable (retrieval, chunking, embedding, reranking, model,
    temperature, seed, relationship expansion) differs -- matches this
    project's established single-variable-isolation pattern for baseline
    configs (see test_classic_rag_baseline_v1_changes_only_temperature_and_seed).
    """
    baseline = load_config("config/experiments/classic-rag-baseline-v1.yaml")
    secure = load_config("config/experiments/secure-rag-baseline-v1.yaml")

    assert baseline.generation.prompt.version == "v2"
    assert secure.generation.prompt.version == "v3"
    assert baseline.security.authorization.enabled is False
    assert secure.security.authorization.enabled is True

    baseline_dict = baseline.model_dump(
        exclude={"generation": {"prompt"}, "security": {"authorization": {"enabled"}}}
    )
    secure_dict = secure.model_dump(
        exclude={"generation": {"prompt"}, "security": {"authorization": {"enabled"}}}
    )
    assert baseline_dict == secure_dict


def test_secure_rag_baseline_v1_auth_disabled_variant_isolates_only_authorization():
    """The auth-disabled variant differs from secure-rag-baseline-v1.yaml only by that flag."""
    secure = load_config("config/experiments/secure-rag-baseline-v1.yaml")
    disabled = load_config("config/experiments/secure-rag-baseline-v1-auth-disabled.yaml")

    assert secure.security.authorization.enabled is True
    assert disabled.security.authorization.enabled is False

    secure_dict = secure.model_dump(exclude={"security": {"authorization": {"enabled"}}})
    disabled_dict = disabled.model_dump(exclude={"security": {"authorization": {"enabled"}}})
    assert secure_dict == disabled_dict


def test_all_live_config_files_migrated_to_new_cutoff_field_names():
    """Every config file AppConfig ever loads uses the new retrieval/reranker cutoff fields.

    Documents the migration policy chosen for the retrieval-cutoff refactor
    (retrieval.top_k -> retrieval.candidate_k, retrieval.rerank_top_n split
    into reranker.top_n + retrieval.generation_context_top_n): the 9 live
    config files (config/default.yaml + config/experiments/*.yaml) were
    migrated explicitly to the new field names rather than supporting the
    old names alongside the new ones. Historical experiments/configs/*.yaml
    snapshots are untouched, unmigrated archival copies -- they are never
    re-parsed by AppConfig (only copied as MLflow artifacts by
    scripts/record_experiment.py), so they're deliberately excluded here.
    """
    config_paths = [
        DEFAULT_CONFIG_PATH,
        *sorted((REPO_ROOT / "config" / "experiments").glob("*.yaml")),
    ]
    assert len(config_paths) >= 9

    for path in config_paths:
        raw_text = path.read_text(encoding="utf-8")
        # Exact-line checks (not a bare substring) -- the auth-boundary
        # milestone's `security.dos_limits.max_top_k` field legitimately
        # contains the substring "top_k:" without being the old
        # retrieval.top_k field this test guards against.
        assert "\n  top_k:" not in raw_text, f"{path} still has the old retrieval.top_k field"
        assert (
            "\n  rerank_top_n:" not in raw_text
        ), f"{path} still has the old retrieval.rerank_top_n field"

        config = load_config(str(path))
        assert config.retrieval.candidate_k > 0
        assert config.retrieval.generation_context_top_n > 0
        assert config.reranker.top_n > 0
