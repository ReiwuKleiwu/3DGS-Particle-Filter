"""Filesystem locations used by the standalone evaluation tools."""

from __future__ import annotations

from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_DIR.parent
ARTIFACTS_DIR = EVALUATION_DIR / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
RESULTS_DIR = ARTIFACTS_DIR / "results"
