"""Converts absolute poses into odometry deltas in the robot frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.particle_filter.domain.pose import Pose2D, wrap_angle


@dataclass(frozen=True)
class OdometryDelta:
    forward_meters: float
    lateral_meters: float
    yaw_radians: float


def compute_odometry_delta_in_robot_frame(previous_pose: Pose2D, current_pose: Pose2D) -> OdometryDelta:
    delta_x_world = current_pose.x - previous_pose.x
    delta_y_world = current_pose.y - previous_pose.y

    previous_yaw = previous_pose.yaw
    cosine = math.cos(previous_yaw)
    sine = math.sin(previous_yaw)

    forward_meters = cosine * delta_x_world + sine * delta_y_world
    lateral_meters = -sine * delta_x_world + cosine * delta_y_world
    yaw_radians = wrap_angle(current_pose.yaw - previous_yaw)

    return OdometryDelta(
        forward_meters=forward_meters,
        lateral_meters=lateral_meters,
        yaw_radians=yaw_radians,
    )
