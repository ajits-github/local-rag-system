"""Unit tests for `rag.eval.run_metadata.capture_git_metadata`.

Regression for EVAL-015: no experiment-recording code path anywhere in
the repo captured which exact commit produced a result.
"""

from __future__ import annotations

import subprocess

from rag.eval.run_metadata import capture_git_metadata


def test_capture_git_metadata_returns_real_values_in_this_repo():
    """Running inside an actual git checkout returns a real commit/branch, not None."""
    metadata = capture_git_metadata()

    assert metadata["git_commit_sha"]
    assert len(metadata["git_commit_sha"]) == 40
    assert metadata["git_branch"]
    assert metadata["git_dirty"] in (True, False)


def test_capture_git_metadata_degrades_to_none_fields_when_git_is_unavailable(monkeypatch):
    """A missing git executable (or any subprocess failure) never raises."""

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    metadata = capture_git_metadata()

    assert metadata == {"git_commit_sha": None, "git_branch": None, "git_dirty": None}
