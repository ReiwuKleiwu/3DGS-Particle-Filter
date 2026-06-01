"""Implements the core particle-filter state machine and update operations."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from core.particle_filter.domain.estimator import estimate_weighted_pose
from core.particle_filter.domain.measurement_update import MeasurementUpdateStats, apply_measurement_update
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.odometry import OdometryDelta
from core.particle_filter.domain.particle import Particle
from core.particle_filter.domain.pose import Pose2D, Pose2DPrior
from core.particle_filter.domain.resampler import SystematicResampler


@dataclass(frozen=True)
class TurtleBotParticleFilterConfig:
    particle_count: int
    resample_threshold_ratio: float = 0.5
    roughening_enabled: bool = True
    roughening_mode: str = "resample_only"
    roughening_ratio: float = 0.0
    roughening_sigma_x: float = 0.05
    roughening_sigma_y: float = 0.05
    roughening_sigma_yaw: float = 0.05


class TurtleBotParticleFilter:
    def __init__(
        self,
        *,
        config: TurtleBotParticleFilterConfig,
        motion_model: TurtleBotMotionModel,
        resampler: SystematicResampler | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._rng = rng or random.Random()
        self._motion_model = motion_model
        self._resampler = resampler or SystematicResampler(rng=self._rng)
        self._particles: list[Particle] = []
        self._last_random_particle_count = 0
        self._last_roughening_particle_count = 0

    @property
    def config(self) -> TurtleBotParticleFilterConfig:
        return self._config

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    @property
    def last_random_particle_count(self) -> int:
        return self._last_random_particle_count

    @property
    def last_roughening_particle_count(self) -> int:
        return self._last_roughening_particle_count

    def initialize(self, prior: Pose2DPrior) -> None:
        uniform_weight = 1.0 / self._config.particle_count
        self._particles = []

        for _ in range(self._config.particle_count):
            sampled_pose = Pose2D(
                x=prior.mean.x + self._rng.gauss(0.0, prior.sigma_x),
                y=prior.mean.y + self._rng.gauss(0.0, prior.sigma_y),
                yaw=prior.mean.yaw + self._rng.gauss(0.0, prior.sigma_yaw),
            )
            self._particles.append(Particle(pose=sampled_pose, weight=uniform_weight))

    def initialize_global(self, pose_sampler: Callable[[], Pose2D]) -> None:
        uniform_weight = 1.0 / self._config.particle_count
        self._particles = [Particle(pose=pose_sampler(), weight=uniform_weight) for _ in range(self._config.particle_count)]

    def reconfigure(
        self,
        *,
        particle_count: int | None = None,
        resample_threshold_ratio: float | None = None,
        roughening_ratio: float | None = None,
        roughening_enabled: bool | None = None,
        roughening_mode: str | None = None,
        roughening_sigma_x: float | None = None,
        roughening_sigma_y: float | None = None,
        roughening_sigma_yaw: float | None = None,
    ) -> None:
        next_particle_count = int(particle_count if particle_count is not None else self._config.particle_count)
        next_resample_ratio = float(
            resample_threshold_ratio if resample_threshold_ratio is not None else self._config.resample_threshold_ratio
        )
        self._config = TurtleBotParticleFilterConfig(
            particle_count=next_particle_count,
            resample_threshold_ratio=next_resample_ratio,
            roughening_enabled=bool(
                roughening_enabled if roughening_enabled is not None else self._config.roughening_enabled
            ),
            roughening_mode=str(roughening_mode if roughening_mode is not None else self._config.roughening_mode),
            roughening_ratio=float(roughening_ratio if roughening_ratio is not None else self._config.roughening_ratio),
            roughening_sigma_x=float(roughening_sigma_x if roughening_sigma_x is not None else self._config.roughening_sigma_x),
            roughening_sigma_y=float(roughening_sigma_y if roughening_sigma_y is not None else self._config.roughening_sigma_y),
            roughening_sigma_yaw=float(
                roughening_sigma_yaw if roughening_sigma_yaw is not None else self._config.roughening_sigma_yaw
            ),
        )

        if self._particles and len(self._particles) != next_particle_count:
            self._particles = self._resampler.resample_to_count(self._particles, next_particle_count)

    def predict_from_odometry(self, odometry_delta: OdometryDelta) -> None:
        self._particles = self._motion_model.predict(self._particles, odometry_delta)

    def update_from_measurement_errors(self, measurement_errors: list[float], *, temperature: float) -> MeasurementUpdateStats:
        return apply_measurement_update(self._particles, measurement_errors, temperature=temperature)

    def effective_particle_count(self) -> float:
        if not self._particles:
            return 0.0
        return 1.0 / sum(particle.weight * particle.weight for particle in self._particles)

    def resample_if_needed(
        self,
        *,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
        random_particle_ratio: float = 0.0,
    ) -> bool:
        if not self._particles:
            return False

        threshold = self._config.resample_threshold_ratio * len(self._particles)
        should_force_recovery_resample = random_pose_sampler is not None and random_particle_ratio > 0.0
        if self.effective_particle_count() >= threshold and not should_force_recovery_resample:
            self._last_random_particle_count = 0
            self._last_roughening_particle_count = 0
            return False

        self._particles = self._resampler.resample(
            self._particles,
            random_pose_sampler=random_pose_sampler,
            random_particle_ratio=random_particle_ratio,
            roughening_ratio=(
                self._config.roughening_ratio
                if self._config.roughening_enabled and self._config.roughening_mode == "resample_only"
                else 0.0
            ),
            roughening_sigma_x=self._config.roughening_sigma_x,
            roughening_sigma_y=self._config.roughening_sigma_y,
            roughening_sigma_yaw=self._config.roughening_sigma_yaw,
        )
        self._last_random_particle_count = self._resampler.last_random_count
        self._last_roughening_particle_count = self._resampler.last_roughening_count
        return True

    def roughen_always_if_configured(self) -> bool:
        if (
            not self._particles
            or not self._config.roughening_enabled
            or self._config.roughening_mode != "always"
            or self._config.roughening_ratio <= 0.0
        ):
            return False
        self._particles = self._resampler.roughen(
            self._particles,
            roughening_ratio=self._config.roughening_ratio,
            roughening_sigma_x=self._config.roughening_sigma_x,
            roughening_sigma_y=self._config.roughening_sigma_y,
            roughening_sigma_yaw=self._config.roughening_sigma_yaw,
        )
        self._last_roughening_particle_count = self._resampler.last_roughening_count
        return self._last_roughening_particle_count > 0

    def estimate_pose(self) -> Pose2D:
        return estimate_weighted_pose(self._particles)
