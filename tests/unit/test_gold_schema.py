from __future__ import annotations

from pathlib import Path

from rag.eval.gold_schema import (
    GoldExample,
    load_gold_jsonl,
    normalize_for_match,
    reference_context_is_supported,
    source_matches_relevant,
)


def test_load_gold_jsonl_parses_each_line(tmp_path: Path):
    """Each JSONL line becomes a GoldExample with its fields populated."""
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"question": "what?", "relevant_documents": ["knowledge_base/a.md"]}\n'
        '{"question": "huh?", "unanswerable": true}\n',
        encoding="utf-8",
    )

    examples = load_gold_jsonl(path)

    assert len(examples) == 2
    assert examples[0].question == "what?"
    assert examples[0].relevant_documents == ["knowledge_base/a.md"]
    assert examples[1].unanswerable is True


def test_load_gold_jsonl_skips_blank_lines(tmp_path: Path):
    """Blank lines in the JSONL file are skipped, not parsed as examples."""
    path = tmp_path / "gold.jsonl"
    path.write_text('{"question": "what?"}\n\n\n', encoding="utf-8")

    examples = load_gold_jsonl(path)

    assert len(examples) == 1


def test_agentic_fields_default_to_false_and_empty_so_old_gold_files_still_parse():
    """The 7 agentic-milestone fields are all optional/defaulted -- additive, not breaking."""
    example = GoldExample(question="what?")
    assert example.requires_query_decomposition is False
    assert example.requires_multiple_retrieval_calls is False
    assert example.requires_latest_document_tool is False
    assert example.expects_insufficient_evidence_retry is False
    assert example.tool_not_needed is False
    assert example.expects_max_step_termination is False
    assert example.adversarial_tool_instruction is False
    assert example.expected_tool_sequence == []


def test_agentic_fields_parse_when_present():
    """The 7 agentic-milestone fields parse correctly when a gold row supplies them."""
    example = GoldExample.model_validate(
        {
            "question": "which service caused the backlog, and what is the rollback?",
            "requires_query_decomposition": True,
            "requires_multiple_retrieval_calls": True,
            "expected_tool_sequence": ["search_knowledge_base", "get_related_context"],
        }
    )
    assert example.requires_query_decomposition is True
    assert example.requires_multiple_retrieval_calls is True
    assert example.expected_tool_sequence == ["search_knowledge_base", "get_related_context"]


def test_gold_example_defaults():
    """A GoldExample with only a question gets sane field defaults."""
    example = GoldExample(question="what?")
    assert example.expected_answer is None
    assert example.relevant_documents == []
    assert example.unanswerable is False
    assert example.content_type is None
    assert example.reference_contexts == []
    assert example.reference_visual_contexts == []
    assert example.relevant_images == []
    assert example.relevant_sections == []
    assert example.requires_vision is False
    assert example.requires_relationship_expansion is False
    assert example.safety_category is None
    assert example.user_tenant is None
    assert example.user_roles == []
    assert example.allowed_documents == []
    assert example.forbidden_documents == []
    assert example.expected_behavior is None
    assert example.requires_current_document is False
    assert example.expected_document_version is None
    assert example.query_as_of is None
    assert example.injection_present is False
    assert example.injection_source is None
    assert example.sensitive_data_present is False
    assert example.requires_authorization_filter is False
    assert example.requires_tenant_filter is False
    assert example.requires_trust_filter is False
    assert example.expected_trust_level is None


def test_safety_freshness_fields_parse():
    """A gold row with every new safety/freshness field populates them correctly."""
    example = GoldExample.model_validate(
        {
            "question": "As a Beta administrator, what is Alpha's callback route?",
            "safety_category": "cross_tenant_access",
            "user_tenant": "tenant_beta",
            "user_roles": ["tenant_beta_admin"],
            "allowed_documents": ["knowledge_base/governance/authorization-matrix.md"],
            "forbidden_documents": ["knowledge_base/tenant_alpha/confidential-runbook.md"],
            "expected_behavior": "refuse_unauthorized",
            "requires_current_document": False,
            "expected_document_version": None,
            "query_as_of": "2026-08-14",
            "injection_present": False,
            "injection_source": None,
            "sensitive_data_present": True,
            "requires_authorization_filter": True,
            "requires_tenant_filter": True,
            "requires_trust_filter": False,
            "expected_trust_level": None,
        }
    )

    assert example.safety_category == "cross_tenant_access"
    assert example.user_tenant == "tenant_beta"
    assert example.user_roles == ["tenant_beta_admin"]
    assert example.forbidden_documents == ["knowledge_base/tenant_alpha/confidential-runbook.md"]
    assert example.expected_behavior == "refuse_unauthorized"
    assert example.query_as_of == "2026-08-14"
    assert example.sensitive_data_present is True
    assert example.requires_authorization_filter is True


def test_old_schema_gold_file_still_parses(tmp_path: Path):
    """A gold file with none of the new multimodal fields still parses (backward compatibility)."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        '{"question": "what?", "expected_answer": "an answer", '
        '"relevant_documents": ["knowledge_base/a.md"], "question_type": "single_document", '
        '"difficulty": "easy", "unanswerable": false}\n',
        encoding="utf-8",
    )

    examples = load_gold_jsonl(path)

    assert len(examples) == 1
    assert examples[0].requires_vision is False
    assert examples[0].reference_contexts == []


def test_new_schema_fields_parse():
    """A gold row with every new multimodal field populates them correctly."""
    example = GoldExample.model_validate(
        {
            "question": "What P95 latency is shown at 14:00?",
            "expected_answer": "420 ms.",
            "relevant_documents": ["knowledge_base/operations/api-performance-review.md"],
            "content_type": "image_only",
            "requires_vision": True,
            "requires_relationship_expansion": False,
            "relevant_images": ["knowledge_base/operations/images/api-latency-by-hour.png"],
            "relevant_sections": [],
            "reference_contexts": [],
            "reference_visual_contexts": [
                "The API latency chart shows P95 latency at 14:00 as 420 ms."
            ],
        }
    )

    assert example.content_type == "image_only"
    assert example.requires_vision is True
    assert example.relevant_images == ["knowledge_base/operations/images/api-latency-by-hour.png"]
    assert example.reference_visual_contexts == [
        "The API latency chart shows P95 latency at 14:00 as 420 ms."
    ]


def test_normalize_for_match_collapses_whitespace_and_lowercases():
    """normalize_for_match smooths line-wraps/case, not semantics."""
    assert normalize_for_match("Retry  Lock\nTTL") == "retry lock ttl"


def test_reference_context_is_supported_finds_verbatim_substring():
    """A reference_contexts entry matches when it's a normalized substring of a candidate."""
    candidates = ['Some prose.\n\n```json\n{"retry_lock_ttl_seconds": 600}\n```\n\nMore prose.']
    assert reference_context_is_supported('{"retry_lock_ttl_seconds": 600}', candidates)


def test_reference_context_is_supported_rejects_paraphrase():
    """A paraphrased reference (not a verbatim excerpt) is correctly reported as unsupported."""
    candidates = ["The retry lock lasts ten minutes before expiring."]
    assert not reference_context_is_supported(
        '{"retry_lock_ttl_seconds": 600}',
        candidates,
    )


def test_reference_context_is_supported_false_for_blank_reference():
    """A blank/whitespace-only reference never counts as supported."""
    assert not reference_context_is_supported("   ", ["anything at all"])


def test_source_matches_relevant_ignores_root_prefix():
    """A relevant path matches a stored source regardless of its root prefix."""
    assert source_matches_relevant(
        "data/knowledge_base/security/access-control-policy.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_handles_windows_backslashes():
    """Backslash-separated stored sources are matched the same as POSIX ones."""
    assert source_matches_relevant(
        "data\\knowledge_base\\security\\access-control-policy.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_rejects_different_file():
    """A different filename under the same directory does not match."""
    assert not source_matches_relevant(
        "data/knowledge_base/security/data-encryption.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_rejects_partial_segment_match():
    """A partial directory-name match (substring, not full segment) is rejected."""
    # "hitecture/system-overview.md" must not match ".../architecture/system-overview.md"
    assert not source_matches_relevant(
        "data/knowledge_base/architecture/system-overview.md",
        "hitecture/system-overview.md",
    )


def test_source_matches_relevant_rejects_when_relevant_path_longer():
    """A relevant path with more segments than the stored source cannot match."""
    assert not source_matches_relevant(
        "system-overview.md",
        "knowledge_base/architecture/system-overview.md",
    )
