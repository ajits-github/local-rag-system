from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from rag.config import load_config
from rag.prompts.loader import (
    PromptTemplate,
    load_prompt_template,
    load_prompt_template_from_config,
)

V1_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "rag"
    / "prompts"
    / "templates"
    / "rag_answer_v1.yaml"
)

# Byte-identical to the pre-versioning hardcoded _PROMPT_TEMPLATE constant
# that lived in retrieval/pipeline.py -- frozen here (not imported, since
# that constant no longer exists) so v1's rendered output can be pinned.
_LEGACY_PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {query}

Answer:"""


def test_load_prompt_template_loads_valid_v1_yaml():
    """load_prompt_template parses rag_answer_v1.yaml into a valid PromptTemplate."""
    template = load_prompt_template(V1_PATH)
    assert template.prompt_id == "rag_answer"
    assert template.version == "v1"
    assert template.required_variables == ["context", "query"]


def test_load_prompt_template_missing_file_raises_with_path(tmp_path: Path):
    """A missing prompt file raises FileNotFoundError naming the attempted path."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        load_prompt_template(missing)


def test_load_prompt_template_rejects_undeclared_placeholder(tmp_path: Path):
    """A template placeholder missing from required_variables raises ValueError."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "prompt_id: test\n"
        "version: v1\n"
        "description: test\n"
        'system_template: ""\n'
        "user_template: |-\n"
        "  {context} {stray}\n"
        "required_variables:\n"
        "  - context\n"
        'created_at: "2026-08-06"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stray"):
        load_prompt_template(path)


def test_render_missing_required_variable_raises():
    """render() raises ValueError naming a missing required variable."""
    template = load_prompt_template(V1_PATH)
    with pytest.raises(ValueError, match="query"):
        template.render(context="x")


def test_render_unknown_variable_raises():
    """render() raises ValueError naming an unexpected variable."""
    template = load_prompt_template(V1_PATH)
    with pytest.raises(ValueError, match="bogus"):
        template.render(context="x", query="y", bogus="z")


def test_render_v1_matches_legacy_prompt_byte_for_byte():
    """v1's empty system_template and rendered user text match the old hardcoded prompt."""
    template = load_prompt_template(V1_PATH)
    system, user = template.render(context="CTX", query="Q?")
    assert system == ""
    assert user == _LEGACY_PROMPT_TEMPLATE.format(context="CTX", query="Q?")


def test_load_prompt_template_from_config_version_mismatch_raises():
    """A config declaring a different version than the file's own raises ValueError."""
    config = load_config().model_copy(deep=True)
    config.generation.prompt.version = "v99"
    with pytest.raises(ValueError, match="v99"):
        load_prompt_template_from_config(config)


def test_load_prompt_template_from_config_loads_configured_v1():
    """load_prompt_template_from_config loads the default config's active v1 template."""
    config = load_config()
    template = load_prompt_template_from_config(config)
    assert isinstance(template, PromptTemplate)
    assert template.version == "v1"


def test_prompt_checksum_is_deterministic():
    """sha256 of the same prompt file computed twice is identical."""
    first = hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    second = hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    assert first == second
