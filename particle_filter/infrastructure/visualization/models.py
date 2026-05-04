from __future__ import annotations

from dataclasses import dataclass

from particle_filter.domain.pose import Pose2D


@dataclass(frozen=True)
class VisualizationParticle:
    x: float
    y: float
    yaw: float
    weight: float


@dataclass(frozen=True)
class VisualizationFilterState:
    particle_count: int
    resample_threshold_ratio: float
    temperature: float
    motion_noise_x_meters: float
    motion_noise_y_meters: float
    motion_noise_yaw_radians: float
    paused: bool


@dataclass(frozen=True)
class VisualizationSnapshot:
    update_index: int
    image_stamp_seconds: int
    image_stamp_nanoseconds: int
    particles: list[VisualizationParticle]
    estimated_pose: Pose2D
    ground_truth_pose: Pose2D | None
    best_particle_index: int
    best_score: float
    effective_particle_count: float
    render_and_score_milliseconds: float
    resampled: bool
    observation_image_rgb: object
    best_render_png_bytes: bytes
    filter_state: VisualizationFilterState
