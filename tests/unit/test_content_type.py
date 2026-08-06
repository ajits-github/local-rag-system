from __future__ import annotations

from pathlib import Path

from rag.config import load_config
from rag.eval.content_type import build_document_content_types, classify_example
from rag.eval.gold_schema import GoldExample


def _example(**kwargs) -> GoldExample:
    """Build a minimal GoldExample, defaulting question/relevant_documents."""
    kwargs.setdefault("question", "What?")
    kwargs.setdefault("relevant_documents", ["docs/a.md"])
    return GoldExample(**kwargs)


def test_unanswerable_takes_priority_over_everything():
    """unanswerable=True always classifies as 'unanswerable', regardless of content type."""
    example = _example(unanswerable=True, question_type="multi_hop")
    document_content_types = {"docs/a.md": {"table"}}

    assert classify_example(example, document_content_types) == "unanswerable"


def test_multi_hop_takes_priority_over_content_type():
    """question_type='multi_hop' classifies as 'multi_hop' even if the document is a table."""
    example = _example(question_type="multi_hop")
    document_content_types = {"docs/a.md": {"table"}}

    assert classify_example(example, document_content_types) == "multi_hop"


def test_table_content_type_bucket():
    """A document whose only content_type is 'table' classifies as 'table'."""
    example = _example()
    document_content_types = {"docs/a.md": {"table"}}

    assert classify_example(example, document_content_types) == "table"


def test_code_and_configuration_both_map_to_code_configuration_bucket():
    """Both 'code' and 'configuration' content types map to the 'code_configuration' bucket."""
    example_code = _example()
    example_config = _example()

    assert classify_example(example_code, {"docs/a.md": {"code"}}) == "code_configuration"
    assert (
        classify_example(example_config, {"docs/a.md": {"configuration"}}) == "code_configuration"
    )


def test_no_relevant_document_content_types_defaults_to_prose():
    """A document with no structured content types (or not found at all) classifies as 'prose'."""
    example = _example()

    assert classify_example(example, {"docs/a.md": {"prose"}}) == "prose"
    assert classify_example(example, {}) == "prose"


def test_table_and_chart_disambiguated_by_chart_keyword():
    """A document mixing 'table' and 'chart' picks 'chart' when the question mentions it."""
    document_content_types = {"docs/a.md": {"table", "chart"}}

    chart_question = _example(question="What does the chart caption say about growth?")
    assert classify_example(chart_question, document_content_types) == "chart"

    plain_question = _example(question="What is the value in the second row?")
    assert classify_example(plain_question, document_content_types) == "table"


def test_build_document_content_types_skips_missing_files(tmp_path: Path):
    """A relevant_documents entry with no file on disk is silently skipped, not an error."""
    config = load_config()

    result = build_document_content_types(tmp_path, ["does/not/exist.md"], config)

    assert result == {}


def test_build_document_content_types_reflects_real_chunker_output(tmp_path: Path):
    """A real Markdown file with a table produces {'table'} (plus 'prose' for surrounding text)."""
    (tmp_path / "doc.md").write_text(
        "# Title\n\nIntro text.\n\n| Name | Value |\n|---|---|\n| a | 1 |\n",
        encoding="utf-8",
    )
    config = load_config()

    result = build_document_content_types(tmp_path, ["doc.md"], config)

    assert result["doc.md"] == {"prose", "table"}
