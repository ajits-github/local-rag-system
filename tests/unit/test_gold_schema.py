from __future__ import annotations

from pathlib import Path

from rag.eval.gold_schema import GoldExample, load_gold_jsonl, source_matches_relevant


def test_load_gold_jsonl_parses_each_line(tmp_path: Path):
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
    path = tmp_path / "gold.jsonl"
    path.write_text('{"question": "what?"}\n\n\n', encoding="utf-8")

    examples = load_gold_jsonl(path)

    assert len(examples) == 1


def test_gold_example_defaults():
    example = GoldExample(question="what?")
    assert example.expected_answer is None
    assert example.relevant_documents == []
    assert example.unanswerable is False


def test_source_matches_relevant_ignores_root_prefix():
    assert source_matches_relevant(
        "data/knowledge_base/security/access-control-policy.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_handles_windows_backslashes():
    assert source_matches_relevant(
        "data\\knowledge_base\\security\\access-control-policy.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_rejects_different_file():
    assert not source_matches_relevant(
        "data/knowledge_base/security/data-encryption.md",
        "knowledge_base/security/access-control-policy.md",
    )


def test_source_matches_relevant_rejects_partial_segment_match():
    # "hitecture/system-overview.md" must not match ".../architecture/system-overview.md"
    assert not source_matches_relevant(
        "data/knowledge_base/architecture/system-overview.md",
        "hitecture/system-overview.md",
    )


def test_source_matches_relevant_rejects_when_relevant_path_longer():
    assert not source_matches_relevant(
        "system-overview.md",
        "knowledge_base/architecture/system-overview.md",
    )
