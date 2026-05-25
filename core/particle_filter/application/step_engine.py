"""Shared localization step engine used by both live and offline workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.config.models import MeasurementSettings
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter
from core.particle_filter.domain.pose import Pose2D
from core.particle_filter.domain.recovery import AugmentedMclRecoveryTracker
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererScoreResult, RendererServiceClient
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation
from core.particle_filter.domain.odometry import compute_odometry_delta_in_robot_frame


@dataclass(frozen=True)
class LocalizationStepResult:
    estimated_pose: Pose2D
    effective_particle_count: float
    resampled: bool
    score_result: RendererScoreResult
    best_particle_pose: Pose2D
    best_score: float
    previous_odometry_pose: Pose2D | None
    measurement_likelihood: float
    random_particle_ratio: float


class LocalizationStepEngine:
    """Applies one localization update from an observation against an existing particle filter."""

    def __init__(self, renderer_client: RendererServiceClient, measurement: MeasurementSettings) -> None:
        self._renderer_client = renderer_client
        self._measurement = measurement

    @property
    def measurement(self) -> MeasurementSettings:
        return self._measurement

    def set_measurement(self, measurement: MeasurementSettings) -> None:
        """Updates the measurement settings used for subsequent localization steps."""
        self._measurement = measurement

    def run_step(
        self,
        *,
        particle_filter: TurtleBotParticleFilter,
        observation: TurtleBotObservation,
        previous_odometry_pose: Pose2D | None,
        recovery_tracker: AugmentedMclRecoveryTracker | None = None,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
    ) -> LocalizationStepResult:
        """Runs one predict-score-update-resample-estimate cycle for the given observation."""
        current_odometry_pose = observation.odometry_pose
        if previous_odometry_pose is not None and current_odometry_pose is not None:
            odometry_delta = compute_odometry_delta_in_robot_frame(previous_odometry_pose, current_odometry_pose)
            particle_filter.predict_from_odometry(odometry_delta)

        score_result = self._renderer_client.score_particles(
            particle_poses=[particle.pose for particle in particle_filter.particles],
            observation=observation,
            metric_name=self._measurement.metric_name,
            packed=self._measurement.packed,
            radius_clip=self._measurement.radius_clip,
            hybrid_ssim_weight=self._measurement.hybrid_ssim_weight,
            hybrid_l1_weight=self._measurement.hybrid_l1_weight,
            hybrid_gradient_weight=self._measurement.hybrid_gradient_weight,
            lpips_top_k=self._measurement.lpips_top_k,
            lpips_weight=self._measurement.lpips_weight,
            lpips_net=self._measurement.lpips_net,
        )
        best_particle_pose = particle_filter.particles[score_result.best_index].pose

        update_stats = particle_filter.update_from_measurement_errors(
            score_result.errors,
            temperature=self._measurement.temperature,
        )
        random_particle_ratio = 0.0
        if recovery_tracker is not None and random_pose_sampler is not None:
            random_particle_ratio = recovery_tracker.update(update_stats.measurement_likelihood)
        effective_particle_count = particle_filter.effective_particle_count()
        resampled = particle_filter.resample_if_needed(
            random_pose_sampler=random_pose_sampler,
            random_particle_ratio=random_particle_ratio,
        )
        estimated_pose = particle_filter.estimate_pose()

        return LocalizationStepResult(
            estimated_pose=estimated_pose,
            effective_particle_count=effective_particle_count,
            resampled=resampled,
            score_result=score_result,
            best_particle_pose=best_particle_pose,
            best_score=score_result.errors[score_result.best_index],
            previous_odometry_pose=current_odometry_pose,
            measurement_likelihood=update_stats.measurement_likelihood,
            random_particle_ratio=random_particle_ratio,
        )
