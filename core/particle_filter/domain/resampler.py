"""Provides systematic resampling for weighted particle sets."""

from __future__ import annotations

import random
from collections.abc import Callable

from core.particle_filter.domain.particle import Particle
from core.particle_filter.domain.pose import Pose2D


class SystematicResampler:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def resample(
        self,
        particles: list[Particle],
        *,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
        random_particle_ratio: float = 0.0,
    ) -> list[Particle]:
        return self.resample_to_count(
            particles,
            len(particles),
            random_pose_sampler=random_pose_sampler,
            random_particle_ratio=random_particle_ratio,
        )

    def resample_to_count(
        self,
        particles: list[Particle],
        particle_count: int,
        *,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
        random_particle_ratio: float = 0.0,
    ) -> list[Particle]:
        if not particles or particle_count <= 0:
            return []

        random_particle_ratio = min(1.0, max(0.0, float(random_particle_ratio)))
        if random_particle_ratio > 0.0 and random_pose_sampler is None:
            raise ValueError("random_pose_sampler is required when random_particle_ratio is non-zero")

        random_count = 0
        if random_pose_sampler is not None and random_particle_ratio > 0.0:
            random_count = sum(1 for _ in range(particle_count) if self._rng.random() < random_particle_ratio)
        systematic_count = particle_count - random_count

        cumulative_weights: list[float] = []
        running_total = 0.0
        for particle in particles:
            running_total += particle.weight
            cumulative_weights.append(running_total)

        resampled_particles: list[Particle] = []
        if systematic_count > 0:
            step = running_total / systematic_count
            start = self._rng.random() * step
            source_index = 0
            for index in range(systematic_count):
                position = start + index * step
                while source_index < len(cumulative_weights) - 1 and position > cumulative_weights[source_index]:
                    source_index += 1
                source_particle = particles[source_index]
                resampled_particles.append(Particle(pose=source_particle.pose, weight=0.0))

        if random_count > 0 and random_pose_sampler is not None:
            for _ in range(random_count):
                resampled_particles.append(Particle(pose=random_pose_sampler(), weight=0.0))
            self._rng.shuffle(resampled_particles)

        uniform_weight = 1.0 / particle_count
        for particle in resampled_particles:
            particle.weight = uniform_weight
        return resampled_particles
