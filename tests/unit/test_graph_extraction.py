from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rag.graph.extraction import (
    LLMExtractedRelationship,
    LLMExtractionResult,
    _parse_llm_extraction,
    extract_llm_facts,
)


def test_llm_extraction_result_extra_forbid_rejects_unknown_field():
    """An LLM trying to smuggle an extra field (e.g. an 'auth' key) is rejected outright."""
    with pytest.raises(ValidationError):
        LLMExtractionResult.model_validate({"entities": [], "relationships": [], "extra_field": 1})


def test_llm_extracted_relationship_rejects_out_of_allowlist_type():
    """A `relationship_type` not in `RELATIONSHIP_TYPES` is rejected."""
    with pytest.raises(ValidationError):
        LLMExtractedRelationship.model_validate(
            {
                "subject_type": "Service",
                "subject_name": "X",
                "relationship_type": "DROP_TABLE",
                "object_type": "Database",
                "object_name": "Y",
                "confidence": 0.9,
            }
        )


def test_llm_extracted_relationship_rejects_confidence_out_of_range():
    """A `confidence` outside `[0.0, 1.0]` is rejected."""
    with pytest.raises(ValidationError):
        LLMExtractedRelationship.model_validate(
            {
                "subject_type": "Service",
                "subject_name": "X",
                "relationship_type": "DEPENDS_ON",
                "object_type": "Database",
                "object_name": "Y",
                "confidence": 1.5,
            }
        )


def test_parse_llm_extraction_tolerates_prose_wrapping_the_json():
    """JSON wrapped in prose/markdown code fences still parses."""
    raw = (
        "Sure, here is the extraction:\n"
        '```json\n{"entities": [], "relationships": [{"subject_type": "Service", '
        '"subject_name": "Document Service", "relationship_type": "DEPENDS_ON", '
        '"object_type": "Database", "object_name": "PostgreSQL", "confidence": 0.9}]}\n```'
    )
    result = _parse_llm_extraction(raw)
    assert len(result.relationships) == 1
    assert result.relationships[0].object_name == "PostgreSQL"


def test_parse_llm_extraction_raises_on_non_json():
    """Non-JSON text raises a clear `ValueError`."""
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_llm_extraction("I cannot extract anything from this.")


def test_parse_llm_extraction_raises_on_schema_violation():
    """JSON that doesn't match `LLMExtractionResult`'s schema raises."""
    with pytest.raises(ValueError, match="schema validation"):
        _parse_llm_extraction('{"entities": [], "relationships": [], "unexpected": true}')


class _FixedLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system: str, user: str) -> str:
        return self._response

    def health_check(self) -> bool:
        return True


class _FakeTemplate:
    def render(self, **_: Any) -> tuple[str, str]:
        return "system", "user"


def test_extract_llm_facts_returns_empty_result_on_malformed_response_never_raises():
    """A malformed model response degrades to an empty result, not an exception."""
    llm = _FixedLLM("not json at all")
    result = extract_llm_facts(llm, _FakeTemplate(), "some chunk text")
    assert result.entities == []
    assert result.relationships == []


def test_extract_llm_facts_returns_parsed_result_on_valid_response():
    """A well-formed model response parses into the expected relationships."""
    llm = _FixedLLM(
        '{"entities": [], "relationships": [{"subject_type": "Service", '
        '"subject_name": "Document Service", "relationship_type": "DEPENDS_ON", '
        '"object_type": "Database", "object_name": "PostgreSQL", "confidence": 0.95}]}'
    )
    result = extract_llm_facts(llm, _FakeTemplate(), "some chunk text")
    assert len(result.relationships) == 1
    assert result.relationships[0].confidence == 0.95
