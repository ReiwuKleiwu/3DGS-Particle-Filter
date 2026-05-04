from __future__ import annotations

import random

from particle_filter.domain.particle import Particle


class SystematicResampler:
    def resample(self, particles: list[Particle]) -> list[Particle]:
        return self.resample_to_count(particles, len(particles))

    def resample_to_count(self, particles: list[Particle], particle_count: int) -> list[Particle]:
        if not particles or particle_count <= 0:
            return []

        cumulative_weights: list[float] = []
        running_total = 0.0
        for particle in particles:
            running_total += particle.weight
            cumulative_weights.append(running_total)

        resampled_particles: list[Particle] = []
        step = 1.0 / particle_count
        start = random.random() * step
        positions = [start + index * step for index in range(particle_count)]

        source_index = 0
        for position in positions:
            while source_index < len(cumulative_weights) - 1 and position > cumulative_weights[source_index]:
                source_index += 1
            source_particle = particles[source_index]
            resampled_particles.append(
                Particle(
                    pose=source_particle.pose,
                    weight=1.0 / particle_count,
                )
            )

        return resampled_particles
