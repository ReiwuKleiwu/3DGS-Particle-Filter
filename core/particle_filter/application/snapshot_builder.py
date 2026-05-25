"""Builds frontend visualization snapshots from live localization state."""

from __future__ import annotations

from core.particle_filter.application.runtime_state import LocalizationRuntimeState
from core.particle_filter.application.step_engine import LocalizationStepResult
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation
from core.particle_filter.infrastructure.visualization.models import (
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
            observation_image_rgb=observation.image_rgb,
            best_render_png_bytes=step_result.score_result.best_render_png_bytes,
            filter_state=VisualizationFilterState(
                particle_count=runtime_state.particle_filter_config.particle_count,
                resample_threshold_ratio=runtime_state.particle_filter_config.resample_threshold_ratio,
                temperature=runtime_state.measurement.temperature,
                motion_noise_x_meters=runtime_state.motion_noise.x_meters,
                motion_noise_y_meters=runtime_state.motion_noise.y_meters,
                motion_noise_yaw_radians=runtime_state.motion_noise.yaw_radians,
                paused=runtime_state.paused,
                localization_mode=runtime_state.localization_mode,
            ),
        )
