"""Builds frontend visualization snapshots from live localization state."""

from __future__ import annotations

from core.particle_filter.application.runtime_state import LocalizationRuntimeState
from core.particle_filter.application.step_engine import LocalizationStepResult
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation
from core.particle_filter.infrastructure.visualization.models import (
    VisualizationAdaptiveParticleCountState,
    VisualizationFilterState,
    VisualizationParticle,
    VisualizationSnapshot,
)


class LocalizationSnapshotBuilder:
    """Converts live localization runtime state into frontend snapshot DTOs."""

    def build(
        self,
        *,
        runtime_state: LocalizationRuntimeState,
        observation: TurtleBotObservation,
        step_result: LocalizationStepResult,
    ) -> VisualizationSnapshot:
        """Builds the frontend snapshot DTO from the latest live filter state and step result."""
        adaptive_particle_count = runtime_state.adaptive_particle_count_controller.status()
        return VisualizationSnapshot(
            update_index=runtime_state.update_count,
            image_stamp_seconds=observation.image_stamp_seconds,
            image_stamp_nanoseconds=observation.image_stamp_nanoseconds,
            particles=[
                VisualizationParticle(
                    x=particle.pose.x,
                    y=particle.pose.y,
                    yaw=particle.pose.yaw,
                    weight=particle.weight,
                    recovery_sample=particle.recovery_sample,
                    roughening_sample=particle.roughening_sample,
                )
                for particle in runtime_state.particle_filter.particles
            ],
            estimated_pose=step_result.estimated_pose,
            ground_truth_pose=observation.map_pose,
            amcl_pose=observation.amcl_pose,
            best_particle_index=step_result.score_result.best_index,
            best_particle_pose=step_result.best_particle_pose,
            best_score=step_result.best_score,
            effective_particle_count=step_result.effective_particle_count,
            render_and_score_milliseconds=step_result.score_result.elapsed_milliseconds,
            resampled=step_result.resampled,
            random_particle_ratio=step_result.random_particle_ratio,
            random_particle_count=step_result.random_particle_count,
            roughening_particle_count=step_result.roughening_particle_count,
            observation_image_rgb=observation.image_rgb,
            best_render_png_bytes=step_result.score_result.best_render_png_bytes,
            filter_state=VisualizationFilterState(
                particle_count=runtime_state.particle_filter_config.particle_count,
                resample_threshold_ratio=runtime_state.particle_filter_config.resample_threshold_ratio,
                roughening_enabled=runtime_state.particle_filter_config.roughening_enabled,
                roughening_mode=runtime_state.particle_filter_config.roughening_mode,
                roughening_ratio=runtime_state.particle_filter_config.roughening_ratio,
                roughening_sigma_x=runtime_state.particle_filter_config.roughening_sigma_x,
                roughening_sigma_y=runtime_state.particle_filter_config.roughening_sigma_y,
                roughening_sigma_yaw=runtime_state.particle_filter_config.roughening_sigma_yaw,
                temperature=runtime_state.measurement.temperature,
                motion_noise_x_meters=runtime_state.motion_noise.x_meters,
                motion_noise_y_meters=runtime_state.motion_noise.y_meters,
                motion_noise_yaw_radians=runtime_state.motion_noise.yaw_radians,
                recovery_enabled=runtime_state.recovery_tracker.settings.enabled,
                recovery_strategy=runtime_state.recovery_tracker.settings.strategy,
                recovery_random_particle_floor_ratio=runtime_state.recovery_tracker.settings.random_particle_floor_ratio,
                recovery_random_particle_max_ratio=runtime_state.recovery_tracker.settings.random_particle_max_ratio,
                recovery_absolute_score_profiles=runtime_state.recovery_tracker.settings.absolute_score_profiles,
                adaptive_particle_count=VisualizationAdaptiveParticleCountState(
                    enabled=adaptive_particle_count.enabled,
                    min_particle_count=adaptive_particle_count.min_particle_count,
                    medium_particle_count=adaptive_particle_count.medium_particle_count,
                    max_particle_count=adaptive_particle_count.max_particle_count,
                    target_particle_count=adaptive_particle_count.target_particle_count,
                    stable_required_updates=adaptive_particle_count.stable_required_updates,
                    unstable_required_updates=adaptive_particle_count.unstable_required_updates,
                    xy_spread_stable_meters=adaptive_particle_count.xy_spread_stable_meters,
                    xy_spread_unstable_meters=adaptive_particle_count.xy_spread_unstable_meters,
                    yaw_spread_stable_radians=adaptive_particle_count.yaw_spread_stable_radians,
                    yaw_spread_unstable_radians=adaptive_particle_count.yaw_spread_unstable_radians,
                    best_score_stable_threshold=adaptive_particle_count.best_score_stable_threshold,
                    median_score_stable_threshold=adaptive_particle_count.median_score_stable_threshold,
                    stable_update_count=adaptive_particle_count.stable_update_count,
                    unstable_update_count=adaptive_particle_count.unstable_update_count,
                    xy_spread_meters=adaptive_particle_count.xy_spread_meters,
                    yaw_spread_radians=adaptive_particle_count.yaw_spread_radians,
                    last_resize_reason=adaptive_particle_count.last_resize_reason,
                ),
                paused=runtime_state.paused,
                localization_mode=runtime_state.localization_mode,
            ),
        )
