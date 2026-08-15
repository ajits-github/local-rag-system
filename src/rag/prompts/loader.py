"""Loads and validates versioned prompt templates from YAML.

Prompt loading is a small read/validate/render path, so it stays in this
module rather than going through provider factories.
"""

from __future__ import annotations

import string
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from rag.config import AppConfig


def _placeholder_names(text: str) -> set[str]:
    """Return the `{name}` field names referenced in a str.format template."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


class PromptTemplate(BaseModel):
    """A single versioned generation prompt, loaded from a YAML file."""

    prompt_id: str
    version: str
    description: str
    system_template: str
    user_template: str
    required_variables: list[str]
    created_at: str

    @model_validator(mode="after")
    def _required_variables_match_placeholders(self) -> PromptTemplate:
        """Cross-check `required_variables` against `{...}` placeholders in both templates."""
        found = _placeholder_names(self.system_template) | _placeholder_names(self.user_template)
        required = set(self.required_variables)
        undeclared = sorted(found - required)
        unused = sorted(required - found)
        if undeclared or unused:
            problems = []
            if undeclared:
                problems.append(f"placeholder(s) {undeclared} not listed in required_variables")
            if unused:
                problems.append(f"required_variables {unused} not used by any template placeholder")
            raise ValueError(f"Prompt '{self.prompt_id}' ({self.version}): " + "; ".join(problems))
        return self

    def render(self, **variables: str) -> tuple[str, str]:
        """Render the system and user templates against `variables`.

        Parameters
        ----------
        **variables : str
            Must supply exactly the names in `required_variables` — no
            more, no fewer.

        Returns
        -------
        tuple[str, str]
            ``(system, user)`` rendered text. `system` is ``""`` if
            `system_template` is empty.

        Raises
        ------
        ValueError
            If any required variable is missing, or any unexpected
            keyword argument is supplied. The message names every
            offending variable.
        """
        required = set(self.required_variables)
        provided = set(variables)
        missing = sorted(required - provided)
        unknown = sorted(provided - required)
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing required variable(s): {', '.join(missing)}")
            if unknown:
                parts.append(f"unknown variable(s): {', '.join(unknown)}")
            raise ValueError(
                f"Cannot render prompt '{self.prompt_id}' ({self.version}): " + "; ".join(parts)
            )
        system = self.system_template.format(**variables) if self.system_template else ""
        user = self.user_template.format(**variables)
        return system, user


def load_prompt_template(path: str | Path) -> PromptTemplate:
    """Load and validate a prompt template YAML file.

    Parameters
    ----------
    path : str | Path
        Path to a prompt template YAML file.

    Returns
    -------
    PromptTemplate
        The validated template.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file's contents fail schema validation (including the
        required_variables/placeholder cross-check).
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Prompt template file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    try:
        return PromptTemplate.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid prompt template at {resolved}: {exc}") from exc


def load_prompt_template_from_config(config: AppConfig) -> PromptTemplate:
    """Load the prompt template configured by `config.generation.prompt`.

    Validates that the loaded file's own `prompt_id`/`version` match what
    `config.generation.prompt.id`/`.version` declare, catching a stale or
    mismatched `path` early.

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    PromptTemplate
        The validated, configured template.

    Raises
    ------
    FileNotFoundError
        If the configured `path` doesn't exist.
    ValueError
        If the file fails schema validation, or its `prompt_id`/`version`
        don't match `config.generation.prompt`.
    """
    path = config.prompt_template_path()
    template = load_prompt_template(path)
    expected_id = config.generation.prompt.id
    expected_version = config.generation.prompt.version
    if template.prompt_id != expected_id or template.version != expected_version:
        raise ValueError(
            f"Prompt template at {path} declares prompt_id={template.prompt_id!r} "
            f"version={template.version!r}, but config expects "
            f"id={expected_id!r} version={expected_version!r}"
        )
    return template
