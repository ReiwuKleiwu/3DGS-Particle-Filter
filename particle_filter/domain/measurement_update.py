from __future__ import annotations

import math

from particle_filter.domain.particle import Particle


def apply_measurement_update(
    particles: list[Particle],
    measurement_errors: list[float],
    *,
    temperature: float,
) -> None:
    if len(particles) != len(measurement_errors):
        raise ValueError("Measurement error count must match particle count.")
    if temperature <= 0.0:
        raise ValueError("Measurement temperature must be positive.")

    scaled_logits = [(-error / temperature) for error in measurement_errors]
    max_logit = max(scaled_logits)
    unnormalized_weights = [math.exp(logit - max_logit) for logit in scaled_logits]
    total_weight = sum(unnormalized_weights)

    for particle, normalized_weight in zip(particles, unnormalized_weights):
        particle.weight = normalized_weight / total_weight
