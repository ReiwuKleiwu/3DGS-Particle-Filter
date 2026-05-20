"""Shared helpers for evaluating replay-tuning candidates across manifests."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from core.config import TurtleBotLocalizationConfig
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.replay_tuning.evaluator import evaluate_manifest
from core.replay_tuning.models import ReplayManifest, SearchCandidate, TrialResult


ProgressFactory = Callable[[Path], Callable[[dict], None] | None]


def normalize_weights(ssim: float, l1: float, grad: float) -> tuple[float, float, float]:
    """Normalizes hybrid-measurement weights while preserving a sane fallback."""
    total = ssim + l1 + grad
    if total <= 0.0:
        return 0.25, 0.30, 0.45
    return ssim / total, l1 / total, grad / total


def evaluate_candidate_across_manifests(
    *,
    candidate: SearchCandidate,
    manifests: list[ReplayManifest],
    manifest_paths: list[Path],
    renderer_client: RendererServiceClient,
    settings: TurtleBotLocalizationConfig,
    frame_stride: int,
    progress_factory: ProgressFactory | None = None,
) -> dict:
    """Evaluates one search candidate across all manifests and aggregates the result."""
    per_manifest_results = []
    aggregate_objective = 0.0
    aggregate_translation = 0.0
    aggregate_yaw = 0.0
    aggregate_failure = 0.0
    aggregate_elapsed = 0.0

    for manifest_path, manifest in zip(manifest_paths, manifests):
        progress_callback = None if progress_factory is None else progress_factory(manifest_path)
        trial_result: TrialResult = evaluate_manifest(
            manifest,
            candidate,
            renderer_client,
            settings,
            progress_callback=progress_callback,
            frame_stride=frame_stride,
        )
        per_manifest_results.append(
            {
                "manifest": str(manifest_path),
                "summary": asdict(trial_result.summary),
                "case_results": trial_result.case_results,
            }
        )
        aggregate_objective += trial_result.summary.objective
        aggregate_translation += trial_result.summary.mean_translation_error_m
        aggregate_yaw += trial_result.summary.mean_abs_yaw_error_degrees
        aggregate_failure += trial_result.summary.catastrophic_failure_rate
        aggregate_elapsed += trial_result.summary.mean_elapsed_ms

    manifest_count = len(manifests)
    return {
        "candidate": asdict(candidate),
        "aggregate": {
            "objective": aggregate_objective / manifest_count,
            "mean_translation_error_m": aggregate_translation / manifest_count,
            "mean_abs_yaw_error_degrees": aggregate_yaw / manifest_count,
            "catastrophic_failure_rate": aggregate_failure / manifest_count,
            "mean_elapsed_ms": aggregate_elapsed / manifest_count,
        },
        "manifests": per_manifest_results,
    }
