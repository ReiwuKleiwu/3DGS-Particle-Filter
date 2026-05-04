from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(angle_radians: float) -> float:
    return math.atan2(math.sin(angle_radians), math.cos(angle_radians))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Pose2DPrior:
    mean: Pose2D
    sigma_x: float
    sigma_y: float
    sigma_yaw: float
