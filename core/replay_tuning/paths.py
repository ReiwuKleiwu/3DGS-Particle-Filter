"""Shared filesystem locations for replay-tuning code, datasets, and outputs."""

from __future__ import annotations

from pathlib import Path


REPLAY_TUNING_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPLAY_TUNING_ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
RESULTS_DIR = ARTIFACTS_DIR / "results"

