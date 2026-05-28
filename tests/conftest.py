"""
Shared pytest configuration and safety guards.

The autouse `_guard_data_dir` fixture prevents any test from writing into
the real ./data/ directory. If a test tries to, it fails immediately with
a clear message instead of silently corrupting the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Resolve the real data directory once, at the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REAL_DATA_DIR = (_PROJECT_ROOT / "data").resolve()


@pytest.fixture(autouse=True)
def _guard_data_dir(monkeypatch, tmp_path):
    """
    Auto-applied to every test. Snapshots the real data/ directory before
    the test and restores it after, so nothing leaks even if a test is
    careless. Also fails loudly on accidental writes.
    """
    if _REAL_DATA_DIR.exists():
        before_files = {
            p.relative_to(_REAL_DATA_DIR): p.stat().st_mtime
            for p in _REAL_DATA_DIR.rglob("*") if p.is_file()
        }
    else:
        before_files = {}

    yield

    if _REAL_DATA_DIR.exists():
        after_files = {
            p.relative_to(_REAL_DATA_DIR): p.stat().st_mtime
            for p in _REAL_DATA_DIR.rglob("*") if p.is_file()
        }
    else:
        after_files = {}

    added = set(after_files) - set(before_files)
    removed = set(before_files) - set(after_files)
    modified = {
        f for f in set(before_files) & set(after_files)
        if before_files[f] != after_files[f]
    }

    issues = []
    if added:
        issues.append(f"created files: {sorted(map(str, added))}")
    if removed:
        issues.append(f"deleted files: {sorted(map(str, removed))}")
    if modified:
        issues.append(f"modified files: {sorted(map(str, modified))}")

    if issues:
        # cleanup accidentally-created files so the next test run starts clean
        for rel in added:
            try:
                (_REAL_DATA_DIR / rel).unlink()
            except OSError:
                pass
        pytest.fail(
            "Test touched the real data/ directory: "
            + "; ".join(issues)
            + ". Use tmp_path + monkeypatch.chdir(tmp_path) for any test "
              "that calls generators or writes files."
        )