from __future__ import annotations

import math
import random

from particle_filter.domain.odometry import OdometryDelta
from particle_filter.domain.particle import Particle
from particle_filter.domain.pose import Pose2D, wrap_angle


class TurtleBotMotionModel:
    def __init__(self, *, noise_x: float, noise_y: float, noise_yaw: float) -> None:
        self._noise_x = noise_x
        self._noise_y = noise_y
        self._noise_yaw = noise_yaw

    @property
    def noise_x(self) -> float:
        return self._noise_x

    @property
    def noise_y(self) -> float:
        return self._noise_y

    @property
    def noise_yaw(self) -> float:
        return self._noise_yaw

    def set_noise(self, *, noise_x: float, noise_y: float, noise_yaw: float) -> None:
        self._noise_x = noise_x
        self._noise_y = noise_y
        self._noise_yaw = noise_yaw

    def predict(self, particles: list[Particle], odometry_delta: OdometryDelta) -> list[Particle]:
        predicted_particles: list[Particle] = []

        for particle in particles:
            yaw = particle.pose.yaw
            predicted_x = (
                particle.pose.x
                + math.cos(yaw) * odometry_delta.forward_meters
                - math.sin(yaw) * odometry_delta.lateral_meters
                + random.gauss(0.0, self._noise_x)
            )
            predicted_y = (
                particle.pose.y
                + math.sin(yaw) * odometry_delta.forward_meters
                + math.cos(yaw) * odometry_delta.lateral_meters
                + random.gauss(0.0, self._noise_y)
            )
            predicted_yaw = wrap_angle(yaw + odometry_delta.yaw_radians + random.gauss(0.0, self._noise_yaw))

            predicted_particles.append(
                Particle(
                    pose=Pose2D(x=predicted_x, y=predicted_y, yaw=predicted_yaw),
                    weight=particle.weight,
                )
            )

        return predicted_particles
