"""Defines the mutable particle state used by the filter."""

from __future__ import annotations

from dataclasses import dataclass

from core.particle_filter.domain.pose import Pose2D

@dataclass
class Particle:
    pose: Pose2D
    weight: float
