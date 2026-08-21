"""Structured LLM-driven decision points for the agentic RAG graph.

Every decision point renders a versioned prompt template instructing
JSON-only output, calls the existing, unmodified `LLM.generate(system,
user) -> str`, and parses/validates the response against a Pydantic model
here. This is **agentic structured tool dispatch**. `LLM JSON decision ->
Pydantic validation -> trusted Python dispatcher -> tool execution`. And
explicitly *not* provider-native function/tool calling (i.e. not
`ollama.Client.chat(tools=[...])`, even though the installed client
supports that parameter). The LLM only ever produces text; Python is the
only thing that decides which real function runs and with what (validated)
arguments. See `docs/architecture.md`'s "Agentic RAG" section for the full
rationale.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from rag.generation.base import LLM
from rag.prompts.loader import PromptTemplate

T = TypeVar("T", bound=BaseModel)


class ClassifyDecision(BaseModel):
    """`classify_query` node's output: route a question as simple or complex."""

    query_type: Literal["simple", "complex"]
    reasoning: str = ""


class DecomposeDecision(BaseModel):
    """`decompose` node's output: subquestions for a complex query."""

    subquestions: list[str] = Field(default_factory=list)


class ToolSelectionDecision(BaseModel):
    """`select_tool` node's output: which tool to dispatch, and its raw arguments.

    `tool_args` is intentionally untyped here. It is validated against
    the matching `rag.agent.tool_schemas` model (`extra="forbid"`) by the
    graph driver immediately after this decision is parsed, never trusted
    as-is.
    """

    tool_name: Literal[
        "search_knowledge_base", "get_document", "get_latest_document", "get_related_context"
    ]
    tool_args: dict[str, Any] = Field(default_factory=dict)


class EvidenceSufficiencyDecision(BaseModel):
    """`evaluate_evidence` node's output: whether gathered evidence is enough to answer."""

    sufficient: bool
    reformulated_query: str | None = None
    reasoning: str = ""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Extract the first `{...}` substring from `raw`.

    Tolerates a model wrapping its JSON in prose or a markdown code fence
    without attempting to repair malformed JSON itself. That still fails
    as a parse error in `parse_llm_json`.
    """
    match = _JSON_OBJECT_RE.search(raw)
    return match.group(0) if match else raw


def parse_llm_json(raw: str, model: type[T]) -> T:
    """Parse and validate `raw` LLM output as JSON matching `model`.

    Parameters
    ----------
    raw : str
        The LLM's raw text response.
    model : type[T]
        The Pydantic model the parsed JSON must validate against.

    Returns
    -------
    T
        The validated decision.

    Raises
    ------
    ValueError
        If `raw` contains no valid JSON, or it doesn't validate against
        `model`.
    """
    candidate = _extract_json(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response does not match {model.__name__}: {exc}") from exc


def run_decision(
    llm: LLM,
    template: PromptTemplate,
    model: type[T],
    max_retries: int,
    **variables: str,
) -> T | None:
    """Render `template`, call `llm.generate`, and parse the result as `model`.

    On a parse/validation failure, issues up to `max_retries` bounded
    reparse nudges appended to the user turn. Still plain
    `LLM.generate(system, user) -> str` calls, never a
    structured-output-forcing or native tool-calling API. Before giving
    up.

    Parameters
    ----------
    llm : LLM
        The LLM to call.
    template : PromptTemplate
        The decision point's prompt template.
    model : type[T]
        The Pydantic model the response must validate against.
    max_retries : int
        Bounded number of reparse attempts after the first failure
        (`config.agent.max_json_parse_retries`).
    **variables : str
        Template render variables.

    Returns
    -------
    T | None
        The validated decision, or `None` if every attempt failed.
        Callers must handle `None` with a safe default. Never crash the
        request on a decision-parsing failure.
    """
    system, user = template.render(**variables)
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        prompt_user = user
        if attempt > 0:
            prompt_user = (
                f"{user}\n\nYour previous reply was not valid JSON matching the required "
                f"schema ({last_error}). Reply with ONLY the JSON object, no other text."
            )
        raw = llm.generate(system, prompt_user)
        try:
            return parse_llm_json(raw, model)
        except ValueError as exc:
            last_error = str(exc)
    return None
