"""One-off: log every already-recorded experiments/results/*.json into MLflow.

For experiments recorded before MLflow was integrated (or after a fresh
mlruns/ directory is created). Reuses rag.eval.mlflow_logger.log_experiment
directly -- no eval re-run, no re-computation. Attaches each record's own
JSON plus its matching experiments/configs/<id>.yaml as artifacts; the
original raw eval-output JSON (per-question detail) isn't preserved for
older experiments, so it's simply omitted for those.

Usage:
    python scripts/backfill_mlflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from rag.eval.mlflow_logger import log_experiment  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def main() -> None:
    """CLI entrypoint: log every experiments/results/*.json record to MLflow."""
    config = load_config(str(DEFAULT_CONFIG_PATH))
    results_dir = EXPERIMENTS_DIR / "results"
    configs_dir = EXPERIMENTS_DIR / "configs"

    for results_path in sorted(results_dir.glob("*.json")):
        record = json.loads(results_path.read_text(encoding="utf-8"))
        experiment_id = record.get("experiment_id", results_path.stem)
        config_snapshot_path = configs_dir / f"{experiment_id}.yaml"

        run_id = log_experiment(
            record,
            config.mlflow,
            artifact_paths=[results_path, config_snapshot_path],
        )
        print(f"{experiment_id} -> MLflow run {run_id}")


if __name__ == "__main__":
    main()
