"""Offline replay evaluator built on the shared localization step engine."""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from core.config.models import MeasurementSettings, TurtleBotLocalizationConfig
from core.particle_filter.application.step_engine import LocalizationStepEngine
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from core.particle_filter.domain.pose import wrap_angle
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation
from core.replay_tuning.models import (
    DEFAULT_PRIOR_BANK,
    PriorOffset,
    ReplayFrame,
    ReplayManifest,
    SearchCandidate,
    TrialResult,
    TrialSummary,
)


def build_observation(
    manifest: ReplayManifest,
    frame: ReplayFrame,
    sequence_number: int,
) -> TurtleBotObservation:
    """Builds an offline observation object from one replay frame and its image payload."""
    image = Image.open(manifest.resolve_image_path(frame.image_path)).convert("RGB")
    image_rgb = np.asarray(image, dtype=np.uint8)
    return TurtleBotObservation(
        sequence_number=sequence_number,
        image_rgb=image_rgb,
        image_encoding="rgb8",
        image_frame_id="offline_replay",
        image_stamp_seconds=frame.image_stamp_seconds,
        image_stamp_nanoseconds=frame.image_stamp_nanoseconds,
        camera=manifest.camera,
        odometry_pose=frame.odom_pose,
        map_pose=frame.pose,
        amcl_pose=None,
        resolved_tf_time=frame.resolved_tf_time,
        tf_error=frame.tf_error,
    )


def _measurement_for_candidate(
    base_measurement: MeasurementSettings,
    candidate: SearchCandidate,
) -> MeasurementSettings:
    """Overlays tunable candidate parameters onto the base measurement configuration."""
    return MeasurementSettings(
        metric_name=base_measurement.metric_name,
        temperature=candidate.temperature,
        packed=base_measurement.packed,
        radius_clip=base_measurement.radius_clip,
        hybrid_ssim_weight=candidate.hybrid_ssim_weight,
        hybrid_l1_weight=candidate.hybrid_l1_weight,
        hybrid_gradient_weight=candidate.hybrid_gradient_weight,
        lpips_top_k=candidate.lpips_top_k,
        lpips_weight=candidate.lpips_weight,
        lpips_net=base_measurement.lpips_net,
    )


def evaluate_manifest(
    manifest: ReplayManifest,
    candidate: SearchCandidate,
    renderer_client: RendererServiceClient,
    settings: TurtleBotLocalizationConfig,
    prior_bank: list[PriorOffset] | None = None,
    progress_callback=None,
    frame_stride: int = 1,
) -> TrialResult:
    """Evaluates one parameter candidate over one replay manifest and summarizes the outcome."""
    prior_bank = prior_bank or DEFAULT_PRIOR_BANK
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    sampled_frames = manifest.frames[::frame_stride]
    if not sampled_frames:
        raise ValueError("No frames selected after applying frame_stride")
    case_results: list[dict] = []
    translation_errors: list[float] = []
    yaw_errors_deg: list[float] = []
    elapsed_ms_values: list[float] = []
    failures = 0

    measurement = _measurement_for_candidate(settings.measurement, candidate)

    for case_index, prior_offset in enumerate(prior_bank):
        if progress_callback is not None:
            progress_callback(
                {
                    "type": "case_start",
                    "case_index": case_index,
                    "case_count": len(prior_bank),
                    "frame_count": len(sampled_frames),
                    "prior_offset": asdict(prior_offset),
                }
            )
        prior = prior_offset.apply(
            sampled_frames[0].pose,
            sigma_x=candidate.prior_sigma_x,
            sigma_y=candidate.prior_sigma_y,
            sigma_yaw_degrees=candidate.prior_sigma_yaw_degrees,
        )

        rng = random.Random(candidate.random_seed + case_index)
        motion_model = TurtleBotMotionModel(
            noise_x=candidate.motion_noise_x,
            noise_y=candidate.motion_noise_y,
            noise_yaw=candidate.motion_noise_yaw,
            rng=rng,
        )
        particle_filter = TurtleBotParticleFilter(
            config=TurtleBotParticleFilterConfig(
                particle_count=candidate.particle_count,
                resample_threshold_ratio=candidate.resample_threshold_ratio,
            ),
            motion_model=motion_model,
            rng=rng,
        )
        particle_filter.initialize(prior)
        step_engine = LocalizationStepEngine(renderer_client, measurement)

        previous_odom_pose = None
        last_step_result = None
        total_elapsed_ms = 0.0
        for sequence_number, frame in enumerate(sampled_frames):
            if progress_callback is not None and (
                sequence_number == 0
                or (sequence_number + 1) % 25 == 0
                or sequence_number + 1 == len(sampled_frames)
            ):
                progress_callback(
                    {
                        "type": "frame_progress",
                        "case_index": case_index,
                        "case_count": len(prior_bank),
                        "frame_index": sequence_number + 1,
                        "frame_count": len(sampled_frames),
                    }
                )
            observation = build_observation(manifest, frame, sequence_number)
            last_step_result = step_engine.run_step(
                particle_filter=particle_filter,
                observation=observation,
                previous_odometry_pose=previous_odom_pose,
            )
            previous_odom_pose = last_step_result.previous_odometry_pose
            total_elapsed_ms += last_step_result.score_result.elapsed_milliseconds

        if last_step_result is None:
            raise RuntimeError("Replay evaluation produced no estimate.")

        final_truth = sampled_frames[-1].pose
        last_estimated_pose = last_step_result.estimated_pose
        translation_error = float(np.hypot(last_estimated_pose.x - final_truth.x, last_estimated_pose.y - final_truth.y))
        yaw_error_deg = float(abs(np.rad2deg(wrap_angle(last_estimated_pose.yaw - final_truth.yaw))))
        mean_elapsed_ms = total_elapsed_ms / max(len(sampled_frames), 1)
        failed = translation_error > 0.75 or yaw_error_deg > 25.0
        if failed:
            failures += 1

        translation_errors.append(translation_error)
        yaw_errors_deg.append(yaw_error_deg)
        elapsed_ms_values.append(mean_elapsed_ms)
        case_results.append(
            {
                "case_index": case_index,
                "prior_offset": asdict(prior_offset),
                "translation_error_m": translation_error,
                "yaw_error_degrees": yaw_error_deg,
                "mean_elapsed_ms": mean_elapsed_ms,
                "failed": failed,
                "final_estimate": asdict(last_estimated_pose),
                "final_truth": asdict(final_truth),
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "type": "case_done",
                    "case_index": case_index,
                    "case_count": len(prior_bank),
                    "translation_error_m": translation_error,
                    "yaw_error_degrees": yaw_error_deg,
                    "mean_elapsed_ms": mean_elapsed_ms,
                    "failed": failed,
                }
            )

    summary = TrialSummary(
        mean_translation_error_m=float(statistics.mean(translation_errors)),
        median_translation_error_m=float(statistics.median(translation_errors)),
        max_translation_error_m=float(max(translation_errors)),
        mean_abs_yaw_error_degrees=float(statistics.mean(yaw_errors_deg)),
        catastrophic_failure_rate=float(failures / len(case_results)),
        mean_elapsed_ms=float(statistics.mean(elapsed_ms_values)),
        objective=float(
            statistics.mean(translation_errors)
            + 0.025 * statistics.mean(yaw_errors_deg)
            + 5.0 * (failures / len(case_results))
            + 0.00005 * statistics.mean(elapsed_ms_values)
        ),
    )
    return TrialResult(candidate=candidate, summary=summary, case_results=case_results)


def load_manifests(paths: list[Path]) -> list[ReplayManifest]:
    """Loads a list of replay manifests in the order they were requested."""
    return [ReplayManifest.load(path) for path in paths]
