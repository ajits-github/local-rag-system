"""Unit tests for `rag.ingestion.governance.resolve_ingest_tenant_id`.

Pure-function tests, no DB/pipeline/API needed. Covers the precedence rules
directly: missing governance metadata gets stamped from the caller's own
identity, an explicit same-tenant value is allowed unchanged, a different
tenant is allowed only for a privileged (cross-tenant-support-role) caller,
and rejected otherwise.
"""

from __future__ import annotations

import pytest

from rag.ingestion.governance import (
    IngestCallerContext,
    IngestGovernanceError,
    resolve_ingest_tenant_id,
)


def test_missing_tenant_id_is_stamped_from_caller():
    """No front matter at all (or a PDF/DOCX/HTML loader, which never produces one)."""
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    assert resolve_ingest_tenant_id(None, caller) == "tenant_alpha"


def test_missing_tenant_id_stamped_even_for_privileged_caller():
    """Omitted governance metadata never becomes tenant_id=NULL, privileged or not.

    Per the fix's explicit scope: no broad "global upload" mechanism is
    invented here, so a privileged caller uploading with no front matter
    still gets their own tenant stamped, same as anyone else.
    """
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=True)
    assert resolve_ingest_tenant_id(None, caller) == "tenant_alpha"


def test_explicit_same_tenant_is_allowed_unchanged():
    """Front matter's tenant_id matching the caller's own tenant is a no-op."""
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    assert resolve_ingest_tenant_id("tenant_alpha", caller) == "tenant_alpha"


def test_explicit_different_tenant_rejected_for_normal_caller():
    """A normal (non-privileged) caller cannot assign their upload to another tenant."""
    caller = IngestCallerContext(tenant_id="tenant_alpha", is_privileged=False)
    with pytest.raises(IngestGovernanceError, match="tenant_beta"):
        resolve_ingest_tenant_id("tenant_beta", caller)


def test_explicit_different_tenant_allowed_for_privileged_caller():
    """A caller holding a cross-tenant support role may upload on another tenant's behalf."""
    caller = IngestCallerContext(tenant_id="techfusion_support", is_privileged=True)
    assert resolve_ingest_tenant_id("tenant_beta", caller) == "tenant_beta"


def test_tenant_less_caller_with_no_front_matter_stays_untenanted():
    """A documented, narrow edge case: no claim to stamp from is not a silent broadening.

    Only reachable if a deployment's JWT config drops `tenant_id` from
    `required_claims`, letting a verified-but-tenant-less identity through.
    """
    caller = IngestCallerContext(tenant_id=None, is_privileged=False)
    assert resolve_ingest_tenant_id(None, caller) is None
