from __future__ import annotations

from dataclasses import dataclass

from particle_filter.domain.pose import Pose2D


@dataclass
class Particle:
    pose: Pose2D
    weight: float
