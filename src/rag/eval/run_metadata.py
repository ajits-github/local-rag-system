"""Git commit/branch/dirty-state capture, for experiment reproducibility.

Every experiment record already captures an extremely thorough set of
model/config/corpus facts (see `corpus_lineage.py`, `record_experiment.py`'s
record schema), but none of them identify *which exact version of the
code* produced the result. Given a config snapshot alone, a later reader
cannot tell whether an uncommitted working-tree change was present when a
benchmark ran. `capture_git_metadata()` closes that gap: a best-effort,
non-fatal snapshot of `git rev-parse HEAD`, the current branch, and
whether the working tree was dirty at run time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_git(*args: str) -> str | None:
    """Run one `git` subcommand against `REPO_ROOT`; `None` only on actual failure.

    Never raises: a missing `git` executable, a non-repo checkout (e.g. a
    built/extracted copy with no `.git` directory), or any other failure
    all degrade to `None` rather than failing the eval run that asked for
    this metadata. Returns the (possibly empty) stdout on success, since
    an empty string is a legitimate, meaningful result for some callers
    (`git status --porcelain` on a clean tree), not a failure.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git_metadata() -> dict[str, Any]:
    """Snapshot the repository's current commit/branch/dirty state.

    Returns
    -------
    dict[str, Any]
        ``{"git_commit_sha", "git_branch", "git_dirty"}``. Every field is
        `None` (not raised) when git metadata can't be determined at
        all (e.g. not a git checkout); `git_dirty` is `None` specifically
        (never guessed `False`) when the dirty check itself couldn't run,
        so a reader can distinguish "confirmed clean" from "unknown." A
        clean tree legitimately produces empty `git status --porcelain`
        output, which must not be mistaken for the command having failed.
        `git_commit_sha`/`git_branch` fall back to `None` on an empty
        result too, since an empty value from either is not meaningful.
    """
    commit_sha = _run_git("rev-parse", "HEAD") or None
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD") or None
    status = _run_git("status", "--porcelain")
    dirty = None if status is None else bool(status)
    return {"git_commit_sha": commit_sha, "git_branch": branch, "git_dirty": dirty}
