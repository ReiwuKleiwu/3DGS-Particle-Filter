"""Computes a weighted pose estimate from a set of particles."""

from __future__ import annotations

import math

from core.particle_filter.domain.particle import Particle
from core.particle_filter.domain.pose import Pose2D


def estimate_weighted_pose(particles: list[Particle]) -> Pose2D:
    if not particles:
        raise ValueError("Cannot estimate a pose from an empty particle set.")

    total_weight = sum(particle.weight for particle in particles)
    if total_weight <= 0.0:
        raise ValueError("Particle weights must sum to a positive value.")

    mean_x = sum(particle.pose.x * particle.weight for particle in particles) / total_weight
    mean_y = sum(particle.pose.y * particle.weight for particle in particles) / total_weight

    sine_sum = sum(math.sin(particle.pose.yaw) * particle.weight for particle in particles)
    cosine_sum = sum(math.cos(particle.pose.yaw) * particle.weight for particle in particles)
    mean_yaw = math.atan2(sine_sum, cosine_sum)

    return Pose2D(x=mean_x, y=mean_y, yaw=mean_yaw)
