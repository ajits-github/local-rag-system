"""Tenant, role, trust, and freshness checks for graph facts.

Implements the same rule as `vectorstore.pgvector.build_authorization_where_clause`,
expressed for Cypher (`build_graph_authorization_predicate`) and as a Python
re-check (`is_fact_authorized`). Query templates that take an auth clause
apply the Cypher predicate. Every returned fact, including the shortest-path
template, then passes the Python re-check. The caller's `AuthorizationContext`
must be resolved before this module is used.
"""

from __future__ import annotations

from rag.retrieval.authorization import AuthorizationContext


def build_graph_authorization_predicate(
    auth: AuthorizationContext | None,
    *,
    node_var: str,
    cross_tenant_support_roles: list[str],
) -> tuple[str, dict[str, object]]:
    """Build the tenant/role/trust/freshness `WHERE` fragment for one Cypher node variable.

    A fact is visible when `tenant_id` is NULL, or matches the caller's
    tenant, or the caller holds a cross-tenant support role that is also
    listed in the fact's `allowed_roles`. When `allowed_roles` is set, the
    caller must hold one of those roles. `require_trust_level` and
    `resolved_excluded_document_ids` add further restrictions.

    Parameters
    ----------
    auth : AuthorizationContext | None
        `None` produces an empty (unrestricted) fragment.
    node_var : str
        The Cypher variable name the predicate applies to. Must be a
        hardcoded literal, never caller input.
    cross_tenant_support_roles : list[str]
        Server policy for cross-tenant support access
        (`config.security.authorization.cross_tenant_support_roles`); the
        same list document-level authorization already uses, never a
        second, independently-drifting privilege list.

    Returns
    -------
    tuple[str, dict[str, object]]
        `(cypher_fragment, parameters)`. `cypher_fragment` is `""`
        (unrestricted) or a bare boolean expression referencing `node_var`
        (no leading `WHERE`/`AND`); parameters are namespaced with a
        `_auth_` prefix so they can never collide with a caller-supplied
        query parameter name.
    """
    if auth is None:
        return "", {}

    caller_support_roles = [role for role in auth.roles if role in cross_tenant_support_roles]
    clauses = [
        f"""(
            {node_var}.tenant_id IS NULL
            OR {node_var}.tenant_id = $_auth_tenant_id
            OR (
                {node_var}.allowed_roles IS NOT NULL
                AND any(role IN {node_var}.allowed_roles WHERE role IN $_auth_support_roles)
            )
        )
        AND (
            {node_var}.allowed_roles IS NULL
            OR any(role IN {node_var}.allowed_roles WHERE role IN $_auth_roles)
        )"""
    ]
    parameters: dict[str, object] = {
        "_auth_tenant_id": auth.tenant_id,
        "_auth_support_roles": caller_support_roles,
        "_auth_roles": auth.roles,
    }

    if auth.resolved_excluded_document_ids:
        clauses.append(f"NOT ({node_var}.source_document_id IN $_auth_excluded_document_ids)")
        parameters["_auth_excluded_document_ids"] = auth.resolved_excluded_document_ids

    if auth.require_trust_level:
        clauses.append(
            f"({node_var}.trust_level IS NULL OR {node_var}.trust_level = $_auth_trust_level)"
        )
        parameters["_auth_trust_level"] = auth.require_trust_level

    return " AND ".join(clauses), parameters


def is_fact_authorized(
    properties: dict[str, object],
    auth: AuthorizationContext | None,
    *,
    cross_tenant_support_roles: list[str],
) -> bool:
    """Re-check one returned fact against the same rule in Python.

    This is the final authorization gate for every graph result. It does not
    depend on the query's own `WHERE` clause, so LLM-generated Cypher cannot
    bypass it.

    Parameters
    ----------
    properties : dict[str, object]
        A returned node's or relationship's properties. A missing key is
        treated like `None`, which is unrestricted on that axis.
    auth : AuthorizationContext | None
        `None` means unrestricted; always authorized.
    cross_tenant_support_roles : list[str]
        Same server policy list `build_graph_authorization_predicate` uses.

    Returns
    -------
    bool
        `True` if this fact is authorized for `auth`.
    """
    if auth is None:
        return True

    tenant_id = properties.get("tenant_id")
    allowed_roles = properties.get("allowed_roles")
    trust_level = properties.get("trust_level")
    source_document_id = properties.get("source_document_id")

    caller_support_roles = {role for role in auth.roles if role in cross_tenant_support_roles}
    tenant_ok = (
        tenant_id is None
        or tenant_id == auth.tenant_id
        or (isinstance(allowed_roles, list) and caller_support_roles.intersection(allowed_roles))
    )
    if not tenant_ok:
        return False

    if isinstance(allowed_roles, list) and allowed_roles:
        if not set(auth.roles).intersection(allowed_roles):
            return False

    if (
        auth.resolved_excluded_document_ids
        and source_document_id is not None
        and source_document_id in auth.resolved_excluded_document_ids
    ):
        return False

    if (
        auth.require_trust_level
        and trust_level is not None
        and trust_level != auth.require_trust_level
    ):
        return False

    return True
