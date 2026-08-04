from __future__ import annotations

from pathlib import Path

from rag.eval.gold_schema import GoldExample, load_gold_jsonl


def test_load_gold_jsonl_parses_each_line(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"query_id": "q1", "query": "what?", "relevant_document_ids": ["d1"]}\n'
        '{"query_id": "q2", "query": "huh?", "unanswerable": true}\n',
        encoding="utf-8",
    )

    examples = load_gold_jsonl(path)

    assert len(examples) == 2
    assert examples[0].query_id == "q1"
    assert examples[0].relevant_document_ids == ["d1"]
    assert examples[1].unanswerable is True


def test_load_gold_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    path.write_text('{"query_id": "q1", "query": "what?"}\n\n\n', encoding="utf-8")

    examples = load_gold_jsonl(path)

    assert len(examples) == 1


def test_gold_example_defaults():
    example = GoldExample(query_id="q1", query="what?")
    assert example.relevant_document_ids == []
    assert example.relevant_chunk_ids == []
    assert example.unanswerable is False
