from __future__ import annotations

import random
from dataclasses import dataclass

from particle_filter.domain.estimator import estimate_weighted_pose
from particle_filter.domain.measurement_update import apply_measurement_update
from particle_filter.domain.motion_model import TurtleBotMotionModel
from particle_filter.domain.odometry import OdometryDelta
from particle_filter.domain.particle import Particle
from particle_filter.domain.pose import Pose2D, Pose2DPrior
from particle_filter.domain.resampler import SystematicResampler


@dataclass(frozen=True)
class TurtleBotParticleFilterConfig:
    particle_count: int
    resample_threshold_ratio: float = 0.5


class TurtleBotParticleFilter:
    def __init__(
        self,
        *,
        config: TurtleBotParticleFilterConfig,
        motion_model: TurtleBotMotionModel,
        resampler: SystematicResampler | None = None,
    ) -> None:
        self._config = config
        self._motion_model = motion_model
        self._resampler = resampler or SystematicResampler()
        self._particles: list[Particle] = []

    @property
    def config(self) -> TurtleBotParticleFilterConfig:
        return self._config

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def initialize(self, prior: Pose2DPrior) -> None:
        uniform_weight = 1.0 / self._config.particle_count
        self._particles = []

        for _ in range(self._config.particle_count):
            sampled_pose = Pose2D(
                x=prior.mean.x + random.gauss(0.0, prior.sigma_x),
                y=prior.mean.y + random.gauss(0.0, prior.sigma_y),
                yaw=prior.mean.yaw + random.gauss(0.0, prior.sigma_yaw),
            )
            self._particles.append(Particle(pose=sampled_pose, weight=uniform_weight))

    def reconfigure(
        self,
        *,
        particle_count: int | None = None,
        resample_threshold_ratio: float | None = None,
    ) -> None:
        next_particle_count = int(particle_count if particle_count is not None else self._config.particle_count)
        next_resample_ratio = float(
            resample_threshold_ratio if resample_threshold_ratio is not None else self._config.resample_threshold_ratio
        )
        self._config = TurtleBotParticleFilterConfig(
            particle_count=next_particle_count,
            resample_threshold_ratio=next_resample_ratio,
        )

        if self._particles and len(self._particles) != next_particle_count:
            self._particles = self._resampler.resample_to_count(self._particles, next_particle_count)

    def predict_from_odometry(self, odometry_delta: OdometryDelta) -> None:
        self._particles = self._motion_model.predict(self._particles, odometry_delta)

    def update_from_measurement_errors(self, measurement_errors: list[float], *, temperature: float) -> None:
        apply_measurement_update(self._particles, measurement_errors, temperature=temperature)

    def effective_particle_count(self) -> float:
        if not self._particles:
            return 0.0
        return 1.0 / sum(particle.weight * particle.weight for particle in self._particles)

    def resample_if_needed(self) -> bool:
        if not self._particles:
            return False

        threshold = self._config.resample_threshold_ratio * len(self._particles)
        if self.effective_particle_count() >= threshold:
            return False

        self._particles = self._resampler.resample(self._particles)
        return True

    def estimate_pose(self) -> Pose2D:
        return estimate_weighted_pose(self._particles)
