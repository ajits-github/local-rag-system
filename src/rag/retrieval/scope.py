"""Apply `retrieval.dataset_scope` to a request that may not name a dataset.

Shared by the API routers and the MCP server so both boundaries scope a
missing `dataset_id` the same way.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.config import AppConfig

logger = logging.getLogger(__name__)


class DatasetScopeError(ValueError):
    """Raised when a request names no `dataset_id` and the policy is `reject`."""


def resolve_dataset_id(dataset_id: str | None, config: AppConfig) -> str | None:
    """Return the dataset a request is scoped to, or `None` to search every dataset.

    Parameters
    ----------
    dataset_id : str | None
        The dataset the request named, if any.
    config : AppConfig
        Reads `retrieval.dataset_scope`.

    Returns
    -------
    str | None
        `dataset_id` when given, else `default_dataset_id` when configured,
        else `None` under the `all_datasets` policy.

    Raises
    ------
    DatasetScopeError
        When no dataset is named or configured and the policy is `reject`.
    """
    if dataset_id:
        return dataset_id
    scope = config.retrieval.dataset_scope
    if scope.default_dataset_id:
        return scope.default_dataset_id
    if scope.missing_dataset_policy == "reject":
        raise DatasetScopeError(
            "dataset_id is required: this deployment has several datasets and no default"
        )
    logger.info("request_scoped_to_all_datasets")
    return None


def scope_filters(filters: dict[str, Any] | None, config: AppConfig) -> dict[str, Any] | None:
    """Return `filters` with `dataset_id` filled in or validated per `resolve_dataset_id`.

    Raises
    ------
    DatasetScopeError
        See `resolve_dataset_id`.
    """
    dataset_id = resolve_dataset_id((filters or {}).get("dataset_id"), config)
    if dataset_id is None or (filters or {}).get("dataset_id") == dataset_id:
        return filters
    return {**(filters or {}), "dataset_id": dataset_id}
