"""Model-specific context-window resolution.

`config.generation.model_name` (e.g. `"qwen2.5:1.5b"`) has no built-in
context-window concept anywhere in this codebase today; Ollama itself
defaults every model to a 2048-token context unless `num_ctx` is set
explicitly, and the model's *trained* context length (32K for the
qwen2.5 family) is a separate, larger number than whatever `num_ctx` a
given deployment actually configures. This module makes that an
explicit, configured fact rather than an assumption baked into generation
code.
"""

from __future__ import annotations

from pydantic import BaseModel

from rag.config import AppConfig, ModelContextLimitsConfig


class ModelContextLimits(BaseModel):
    """Resolved context-window facts for the active generation model.

    Attributes
    ----------
    context_window_tokens : int
        Total token budget the model/runtime will accept (prompt +
        completion combined). For Ollama this is `num_ctx`; it is a
        runtime/config fact, not something read from the model file at
        request time, since Ollama exposes no such introspection endpoint.
    max_output_tokens : int
        The hard ceiling on generated tokens, mirrored from
        `config.generation.max_tokens` so the budget planner and the
        actual `num_predict` sent to Ollama can never silently disagree.
    default_reserved_output_tokens : int
        How many tokens the budget planner reserves for generation by
        default (normally equal to `max_output_tokens`, but kept as a
        separate field so a caller can reserve less when it knows the
        expected answer is short, without touching the hard ceiling).
    """

    context_window_tokens: int
    max_output_tokens: int
    default_reserved_output_tokens: int


def resolve_model_context_limits(config: AppConfig) -> ModelContextLimits:
    """Resolve context-window facts for `config.generation.model_name`.

    Looks up `config.generation.model_name` in
    `config.context.model_limits.overrides`; falls back to
    `config.context.model_limits.default_context_window_tokens` when the
    active model has no explicit entry, logging nothing (an unknown model
    falling back to a documented, configurable default is expected, not
    exceptional).

    Parameters
    ----------
    config : AppConfig
        Application configuration.

    Returns
    -------
    ModelContextLimits
        The resolved limits for the currently configured generation model.
    """
    limits_cfg: ModelContextLimitsConfig = config.context.model_limits
    model_name = config.generation.model_name
    context_window = limits_cfg.overrides.get(model_name, limits_cfg.default_context_window_tokens)
    return ModelContextLimits(
        context_window_tokens=context_window,
        max_output_tokens=config.generation.max_tokens,
        default_reserved_output_tokens=config.generation.max_tokens,
    )
