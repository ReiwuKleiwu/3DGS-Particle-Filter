"""Mutable runtime state used by the live localization service."""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.config.models import MeasurementSettings, MotionNoiseSettings
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from core.particle_filter.domain.pose import Pose2D, Pose2DPrior


@dataclass
class LocalizationRuntimeState:
    particle_filter: TurtleBotParticleFilter
    particle_filter_config: TurtleBotParticleFilterConfig
    prior: Pose2DPrior
    motion_noise: MotionNoiseSettings
    measurement: MeasurementSettings
    motion_model: TurtleBotMotionModel
    rng: random.Random
    paused: bool = False
    step_once_requested: bool = False
    previous_odometry_pose: Pose2D | None = None
    last_processed_image_stamp: tuple[int, int] | None = None
    update_count: int = 0
