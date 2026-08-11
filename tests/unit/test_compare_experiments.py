from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    """Import a scripts/*.py module by file path (scripts/ isn't a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_experiments = _load_script("compare_experiments")


def _record(**overrides: object) -> dict:
    """Build a sample experiment record, overriding any fields given."""
    base = {
        "experiment_id": "experiment_001",
        "label": "baseline",
        "timestamp": "2026-08-05T15:42:44+00:00",
        "generation_model": "qwen2.5:1.5b",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_provider": "none",
        "reranker_model": None,
        "recall_at_5": 0.891,
        "recall_at_10": 0.967,
        "hit_rate_at_10": 0.978,
        "mrr": 0.847,
        "answer_quality": 0.432,
        "total_latency_ms": 3736.8,
        "dataset_id": "techfusion",
    }
    base.update(overrides)
    return base


def test_render_table_empty_says_no_experiments():
    """render_table returns a placeholder message for an empty record list."""
    assert "No experiments recorded" in compare_experiments.render_table([])


def test_render_table_includes_row_per_record():
    """render_table formats one Markdown row per experiment record."""
    table = compare_experiments.render_table([_record()])
    expected_row = (
        "| 1 | baseline | dense | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | - | - "
        "| 0.891 | 0.967 | 0.978 | 0.847 | 0.432 | - | - | - | - | 3.7s | techfusion "
        "| 2026-08-05 |"
    )
    assert expected_row in table


def test_render_table_shows_retrieval_provider():
    """render_table shows retrieval_provider, defaulting to 'dense' for pre-hybrid records."""
    table = compare_experiments.render_table([_record(retrieval_provider="hybrid")])
    assert "| baseline | hybrid |" in table

    table_default = compare_experiments.render_table([_record()])
    assert "| baseline | dense |" in table_default


def test_render_table_includes_ragas_columns_when_present():
    """render_table shows ragas_faithfulness/ragas_answer_correctness when a record has them."""
    table = compare_experiments.render_table(
        [_record(ragas_faithfulness=0.786, ragas_answer_correctness=0.469)]
    )
    row = table.splitlines()[-1]
    assert "| 0.786 | 0.469 | 3.7s |" in row


def test_render_table_includes_multimodal_columns_when_present():
    """render_table shows prompt_version/relationship_expansion/supporting-context/image-hit."""
    table = compare_experiments.render_table(
        [
            _record(
                prompt_version="v2",
                relationship_expansion_enabled=True,
                supporting_context_hit_rate=0.788,
                relevant_image_hit_rate=0.842,
            )
        ]
    )
    row = table.splitlines()[-1]
    assert "| v2 | on |" in row
    assert "| 0.788 | 0.842 |" in row


def test_render_table_relationship_expansion_off_shows_off_not_dash():
    """A record with relationship_expansion_enabled=False shows 'off', distinct from '-' (unset)."""
    table = compare_experiments.render_table([_record(relationship_expansion_enabled=False)])
    row = table.splitlines()[-1]
    assert "| off |" in row


def test_render_table_shows_reranker_model_when_present():
    """render_table appends the reranker's model name in parentheses."""
    table = compare_experiments.render_table(
        [
            _record(
                reranker_provider="cross_encoder",
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        ]
    )
    assert "cross_encoder (ms-marco-MiniLM-L-6-v2)" in table


def test_render_table_shortens_embedder_namespace_prefix():
    """render_table drops the embedder's namespace/ prefix for compactness."""
    table = compare_experiments.render_table(
        [_record(embedding_model="sentence-transformers/all-MiniLM-L6-v2")]
    )
    assert "all-MiniLM-L6-v2" in table
    assert "sentence-transformers/all-MiniLM-L6-v2" not in table


def test_short_model_name_handles_missing_value():
    """_short_model_name returns "?" for a missing model name."""
    assert compare_experiments._short_model_name(None) == "?"


def test_render_table_handles_missing_metrics_gracefully():
    """render_table prints "-" for metrics that are None or absent (e.g. pre-RAGAS records)."""
    table = compare_experiments.render_table([_record(answer_quality=None, total_latency_ms=None)])
    row = table.splitlines()[-1]
    expected_row = (
        "| 1 | baseline | dense | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | - | - "
        "| 0.891 | 0.967 | 0.978 | 0.847 | - | - | - | - | - | - | techfusion "
        "| 2026-08-05 |"
    )
    assert row == expected_row


def test_load_records_sorts_by_experiment_id(tmp_path: Path):
    """load_records orders by experiment_id, not filename lexical order."""
    (tmp_path / "experiment_002.json").write_text(
        '{"experiment_id": "experiment_002"}', encoding="utf-8"
    )
    (tmp_path / "experiment_001.json").write_text(
        '{"experiment_id": "experiment_001"}', encoding="utf-8"
    )

    records = compare_experiments.load_records(tmp_path)

    assert [r["experiment_id"] for r in records] == ["experiment_001", "experiment_002"]


def test_load_records_omits_excluded_experiment_ids(tmp_path: Path):
    """load_records drops experiment_ids passed in `exclude` (e.g. a non-comparable pilot)."""
    (tmp_path / "experiment_001.json").write_text(
        '{"experiment_id": "experiment_001"}', encoding="utf-8"
    )
    (tmp_path / "experiment_002.json").write_text(
        '{"experiment_id": "experiment_002"}', encoding="utf-8"
    )

    records = compare_experiments.load_records(tmp_path, exclude={"experiment_002"})

    assert [r["experiment_id"] for r in records] == ["experiment_001"]


def test_update_readme_replaces_content_between_markers(tmp_path: Path, monkeypatch):
    """update_readme swaps the content between the marker comments."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- EXPERIMENTS_TABLE_START -->\nold table\n"
        "<!-- EXPERIMENTS_TABLE_END -->\n\nmore text\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(compare_experiments, "README_PATH", readme)

    updated = compare_experiments.update_readme("new table")

    assert updated is True
    text = readme.read_text(encoding="utf-8")
    assert "new table" in text
    assert "old table" not in text
    assert "more text" in text  # content after the marker is preserved


def test_update_readme_returns_false_when_markers_missing(tmp_path: Path, monkeypatch):
    """update_readme is a no-op returning False when markers aren't found."""
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nno markers here\n", encoding="utf-8")
    monkeypatch.setattr(compare_experiments, "README_PATH", readme)

    assert compare_experiments.update_readme("new table") is False
