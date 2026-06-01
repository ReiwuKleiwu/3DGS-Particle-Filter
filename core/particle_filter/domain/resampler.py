"""Provides systematic resampling for weighted particle sets."""

from __future__ import annotations

import random
from collections.abc import Callable

from core.particle_filter.domain.particle import Particle
from core.particle_filter.domain.pose import Pose2D, wrap_angle


class SystematicResampler:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._last_random_count = 0
        self._last_roughening_count = 0

    @property
    def last_random_count(self) -> int:
        return self._last_random_count

    @property
    def last_roughening_count(self) -> int:
        return self._last_roughening_count

    def resample(
        self,
        particles: list[Particle],
        *,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
        random_particle_ratio: float = 0.0,
        roughening_ratio: float = 0.0,
        roughening_sigma_x: float = 0.0,
        roughening_sigma_y: float = 0.0,
        roughening_sigma_yaw: float = 0.0,
    ) -> list[Particle]:
        return self.resample_to_count(
            particles,
            len(particles),
            random_pose_sampler=random_pose_sampler,
            random_particle_ratio=random_particle_ratio,
            roughening_ratio=roughening_ratio,
            roughening_sigma_x=roughening_sigma_x,
            roughening_sigma_y=roughening_sigma_y,
            roughening_sigma_yaw=roughening_sigma_yaw,
        )

    def resample_to_count(
        self,
        particles: list[Particle],
        particle_count: int,
        *,
        random_pose_sampler: Callable[[], Pose2D] | None = None,
        random_particle_ratio: float = 0.0,
        roughening_ratio: float = 0.0,
        roughening_sigma_x: float = 0.0,
        roughening_sigma_y: float = 0.0,
        roughening_sigma_yaw: float = 0.0,
    ) -> list[Particle]:
        if not particles or particle_count <= 0:
            self._last_random_count = 0
            self._last_roughening_count = 0
            return []

        random_particle_ratio = min(1.0, max(0.0, float(random_particle_ratio)))
        roughening_ratio = min(1.0, max(0.0, float(roughening_ratio)))
        if random_particle_ratio > 0.0 and random_pose_sampler is None:
            raise ValueError("random_pose_sampler is required when random_particle_ratio is non-zero")

        random_count = 0
        if random_pose_sampler is not None and random_particle_ratio > 0.0:
            random_count = sum(1 for _ in range(particle_count) if self._rng.random() < random_particle_ratio)
        self._last_random_count = random_count
        remaining_count = particle_count - random_count
        roughening_count = self._sample_count_from_ratio(remaining_count, roughening_ratio)
        self._last_roughening_count = roughening_count
        systematic_count = remaining_count - roughening_count

        cumulative_weights: list[float] = []
        running_total = 0.0
        for particle in particles:
            running_total += particle.weight
            cumulative_weights.append(running_total)

        resampled_particles: list[Particle] = []
        source_particles = self._systematic_sample_sources(particles, cumulative_weights, running_total, systematic_count)
        roughening_sources = self._systematic_sample_sources(particles, cumulative_weights, running_total, roughening_count)

        for source_particle in source_particles:
            resampled_particles.append(
                Particle(
                    pose=source_particle.pose,
                    weight=0.0,
                    recovery_sample=False,
                    roughening_sample=source_particle.roughening_sample,
                )
            )

        for source_particle in roughening_sources:
            resampled_particles.append(
                Particle(
                    pose=self._jitter_pose(
                        source_particle.pose,
                        sigma_x=roughening_sigma_x,
                        sigma_y=roughening_sigma_y,
                        sigma_yaw=roughening_sigma_yaw,
                    ),
                    weight=0.0,
                    recovery_sample=False,
                    roughening_sample=True,
                )
            )

        if random_count > 0 and random_pose_sampler is not None:
            for _ in range(random_count):
                resampled_particles.append(
                    Particle(
                        pose=random_pose_sampler(),
                        weight=0.0,
                        recovery_sample=True,
                        roughening_sample=False,
                    )
                )
            self._rng.shuffle(resampled_particles)

        uniform_weight = 1.0 / particle_count
        for particle in resampled_particles:
            particle.weight = uniform_weight
        return resampled_particles

    def roughen(
        self,
        particles: list[Particle],
        *,
        roughening_ratio: float,
        roughening_sigma_x: float,
        roughening_sigma_y: float,
        roughening_sigma_yaw: float,
    ) -> list[Particle]:
        if not particles:
            self._last_roughening_count = 0
            return []

        roughening_ratio = min(1.0, max(0.0, float(roughening_ratio)))
        roughening_count = self._sample_count_from_ratio(len(particles), roughening_ratio)
        self._last_roughening_count = roughening_count
        if roughening_count <= 0:
            return [
                Particle(
                    pose=particle.pose,
                    weight=particle.weight,
                    recovery_sample=particle.recovery_sample,
                    roughening_sample=particle.roughening_sample,
                )
                for particle in particles
            ]

        roughening_candidates = [particle for particle in particles if not particle.recovery_sample]
        if not roughening_candidates:
            self._last_roughening_count = 0
            return [
                Particle(
                    pose=particle.pose,
                    weight=particle.weight,
                    recovery_sample=particle.recovery_sample,
                    roughening_sample=particle.roughening_sample,
                )
                for particle in particles
            ]
        roughening_count = min(roughening_count, len(roughening_candidates))
        self._last_roughening_count = roughening_count

        cumulative_weights: list[float] = []
        running_total = 0.0
        for particle in roughening_candidates:
            running_total += particle.weight
            cumulative_weights.append(running_total)

        roughening_sources = self._systematic_sample_sources(
            roughening_candidates,
            cumulative_weights,
            running_total,
            roughening_count,
        )
        roughened_particles = [
            Particle(
                pose=self._jitter_pose(
                    source_particle.pose,
                    sigma_x=roughening_sigma_x,
                    sigma_y=roughening_sigma_y,
                    sigma_yaw=roughening_sigma_yaw,
                ),
                weight=1.0 / len(particles),
                recovery_sample=False,
                roughening_sample=True,
            )
            for source_particle in roughening_sources
        ]

        recovery_particles = [particle for particle in particles if particle.recovery_sample]
        non_recovery_keep_count = len(particles) - len(recovery_particles) - roughening_count
        kept_source_particles = recovery_particles + roughening_candidates[:max(0, non_recovery_keep_count)]
        kept_particles = [
            Particle(
                pose=particle.pose,
                weight=1.0 / len(particles),
                recovery_sample=particle.recovery_sample,
                roughening_sample=False,
            )
            for particle in kept_source_particles
        ]
        next_particles = kept_particles + roughened_particles
        self._rng.shuffle(next_particles)
        return next_particles

    def _systematic_sample_sources(
        self,
        particles: list[Particle],
        cumulative_weights: list[float],
        running_total: float,
        sample_count: int,
    ) -> list[Particle]:
        sampled_sources: list[Particle] = []
        if sample_count > 0:
            step = running_total / sample_count
            start = self._rng.random() * step
            source_index = 0
            for index in range(sample_count):
                position = start + index * step
                while source_index < len(cumulative_weights) - 1 and position > cumulative_weights[source_index]:
                    source_index += 1
                sampled_sources.append(particles[source_index])
        return sampled_sources

    def _sample_count_from_ratio(self, total_count: int, ratio: float) -> int:
        if total_count <= 0 or ratio <= 0.0:
            return 0
        expected_count = total_count * ratio
        count = int(expected_count)
        if self._rng.random() < expected_count - count:
            count += 1
        return min(total_count, max(1, count))

    def _jitter_pose(
        self,
        pose: Pose2D,
        *,
        sigma_x: float,
        sigma_y: float,
        sigma_yaw: float,
    ) -> Pose2D:
        return Pose2D(
            x=pose.x + self._rng.gauss(0.0, max(0.0, float(sigma_x))),
            y=pose.y + self._rng.gauss(0.0, max(0.0, float(sigma_y))),
            yaw=wrap_angle(pose.yaw + self._rng.gauss(0.0, max(0.0, float(sigma_yaw)))),
        )
