"""DTOs used to publish filter snapshots to the frontend service."""

from __future__ import annotations

from dataclasses import dataclass

from core.particle_filter.domain.pose import Pose2D


@dataclass(frozen=True)
class VisualizationParticle:
    x: float
    y: float
    yaw: float
    weight: float
    recovery_sample: bool
    roughening_sample: bool


@dataclass(frozen=True)
class VisualizationFilterState:
    particle_count: int
    resample_threshold_ratio: float
    roughening_enabled: bool
    roughening_mode: str
    roughening_ratio: float
    roughening_sigma_x: float
    roughening_sigma_y: float
    roughening_sigma_yaw: float
    temperature: float
    motion_noise_x_meters: float
    motion_noise_y_meters: float
    motion_noise_yaw_radians: float
    recovery_enabled: bool
    recovery_strategy: str
    recovery_random_particle_floor_ratio: float
    recovery_random_particle_max_ratio: float
    recovery_absolute_score_profiles: dict[str, dict[str, float | int]]
    paused: bool
    localization_mode: str


@dataclass(frozen=True)
class VisualizationSnapshot:
    update_index: int
    image_stamp_seconds: int
    image_stamp_nanoseconds: int
    particles: list[VisualizationParticle]
    estimated_pose: Pose2D
    ground_truth_pose: Pose2D | None
    amcl_pose: Pose2D | None
    best_particle_index: int
    best_particle_pose: Pose2D
    best_score: float
    effective_particle_count: float
    render_and_score_milliseconds: float
    resampled: bool
    random_particle_ratio: float
    random_particle_count: int
    roughening_particle_count: int
    observation_image_rgb: object
    best_render_png_bytes: bytes
    filter_state: VisualizationFilterState
