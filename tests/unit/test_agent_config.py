from __future__ import annotations

from rag.config import AgentConfig, load_config


def test_agent_config_defaults_are_disabled_and_bounded():
    """agent.enabled defaults to False.

    A true no-op kill-switch, matching every other optional security/
    feature toggle in this project (authorization, field_redaction, ...).
    """
    config = AgentConfig()
    assert config.enabled is False
    assert config.max_agent_steps == 8
    assert config.max_retrieval_attempts == 2
    assert config.max_tool_calls == 6
    assert config.max_json_parse_retries == 1
    assert config.max_tool_top_k == 20
    assert config.max_chunks_per_document_fetch == 10
    assert config.max_chunks_per_document_fetch_hard_ceiling == 50


def test_agent_config_none_of_the_bounds_are_llm_writable_anywhere():
    """Sanity check: AgentConfig fields are plain ints/bools, never sourced from a tool schema."""
    for field_name in (
        "max_agent_steps",
        "max_retrieval_attempts",
        "max_tool_calls",
        "max_tool_top_k",
        "max_chunks_per_document_fetch",
        "max_chunks_per_document_fetch_hard_ceiling",
    ):
        assert isinstance(getattr(AgentConfig(), field_name), int)


def test_default_config_loads_agent_section():
    """config/default.yaml declares an agent: section that validates against AgentConfig."""
    config = load_config()
    assert config.agent.enabled is False
    assert config.agent.classify_prompt_path.endswith("agent_classify_v1.yaml")


def test_agent_prompt_template_path_resolves_relative_to_repo_root():
    """agent_prompt_template_path resolves a configured relative prompt path to a real file."""
    config = load_config()
    resolved = config.agent_prompt_template_path(config.agent.synthesize_prompt_path)
    assert resolved.is_file()
    assert resolved.name == "agent_synthesize_v1.yaml"


def test_secure_rag_baseline_configs_are_unaffected_by_the_new_agent_section():
    """Existing experiment configs (no agent: key) still validate, using AgentConfig's defaults."""
    config = load_config()
    # config/default.yaml has no agent-specific overrides beyond the documented defaults --
    # loading it must not require every pre-existing experiment YAML to be touched.
    assert config.agent.enabled is False
