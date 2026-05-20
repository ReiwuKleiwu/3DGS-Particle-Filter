"""Applies measurement likelihoods to particle weights."""

from __future__ import annotations

import math

from core.particle_filter.domain.particle import Particle


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

    prior_weights = [particle.weight for particle in particles]
    if any(weight < 0.0 for weight in prior_weights):
        raise ValueError("Particle weights must be non-negative.")

    scaled_logits = [(-error / temperature) for error in measurement_errors]
    max_logit = max(scaled_logits)
    likelihoods = [math.exp(logit - max_logit) for logit in scaled_logits]
    unnormalized_weights = [
        prior_weight * likelihood
        for prior_weight, likelihood in zip(prior_weights, likelihoods)
    ]
    total_weight = sum(unnormalized_weights)

    if total_weight <= 0.0:
        raise ValueError("Updated particle weights must sum to a positive value.")

    for particle, unnormalized_weight in zip(particles, unnormalized_weights):
        particle.weight = unnormalized_weight / total_weight
