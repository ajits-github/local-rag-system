from __future__ import annotations

from pathlib import Path

from rag.eval.gold_schema import GoldExample, load_gold_jsonl, source_matches_relevant


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


def test_gold_example_defaults():
    """A GoldExample with only a question gets sane field defaults."""
    example = GoldExample(question="what?")
    assert example.expected_answer is None
    assert example.relevant_documents == []
    assert example.unanswerable is False


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
