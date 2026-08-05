from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_experiments = _load_script("compare_experiments")


def _record(**overrides) -> dict:
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
    assert "No experiments recorded" in compare_experiments.render_table([])


def test_render_table_includes_row_per_record():
    table = compare_experiments.render_table([_record()])
    assert "| 1 | baseline | qwen2.5:1.5b | all-MiniLM-L6-v2 | none | 0.891 | 0.967 | 0.978 | 0.847 | 0.432 | 3.7s | techfusion | 2026-08-05 |" in table


def test_render_table_shows_reranker_model_when_present():
    table = compare_experiments.render_table(
        [_record(reranker_provider="cross_encoder", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")]
    )
    assert "cross_encoder (ms-marco-MiniLM-L-6-v2)" in table


def test_render_table_shortens_embedder_namespace_prefix():
    table = compare_experiments.render_table(
        [_record(embedding_model="sentence-transformers/all-MiniLM-L6-v2")]
    )
    assert "all-MiniLM-L6-v2" in table
    assert "sentence-transformers/all-MiniLM-L6-v2" not in table


def test_short_model_name_handles_missing_value():
    assert compare_experiments._short_model_name(None) == "?"


def test_render_table_handles_missing_metrics_gracefully():
    table = compare_experiments.render_table([_record(answer_quality=None, total_latency_ms=None)])
    row = table.splitlines()[-1]
    assert "| - | - |" in row  # answer_quality then total_latency, both missing


def test_load_records_sorts_by_experiment_id(tmp_path: Path):
    (tmp_path / "experiment_002.json").write_text(
        '{"experiment_id": "experiment_002"}', encoding="utf-8"
    )
    (tmp_path / "experiment_001.json").write_text(
        '{"experiment_id": "experiment_001"}', encoding="utf-8"
    )

    records = compare_experiments.load_records(tmp_path)

    assert [r["experiment_id"] for r in records] == ["experiment_001", "experiment_002"]


def test_update_readme_replaces_content_between_markers(tmp_path: Path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- EXPERIMENTS_TABLE_START -->\nold table\n<!-- EXPERIMENTS_TABLE_END -->\n\nmore text\n",
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
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nno markers here\n", encoding="utf-8")
    monkeypatch.setattr(compare_experiments, "README_PATH", readme)

    assert compare_experiments.update_readme("new table") is False
