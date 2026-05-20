"""Shared rendering constants such as camera mounts and map-to-splat alignment."""

from __future__ import annotations

import math


# TurtleBot4 simulator RGB camera intrinsics from /oakd/rgb/preview/camera_info.
SIM_RGB_WIDTH = 320
SIM_RGB_HEIGHT = 240
SIM_RGB_FX = 221.76500408
SIM_RGB_FY = 221.76500408
SIM_RGB_CX = 160.0
SIM_RGB_CY = 120.0


# TurtleBot4 base_link -> oakd_rgb_camera_optical_frame transform, validated from TF/URDF.
# Translation in meters.
TURTLEBOT_RGB_CAMERA_XYZ = (-0.0596, 0.0, 0.24353)

# Orientation expressed explicitly as roll, pitch, yaw relative to base_link.
# This is more readable than hardcoding the equivalent quaternion directly.
TURTLEBOT_RGB_CAMERA_RPY_DEG = (-90.0, 0.0, -90.0)
TURTLEBOT_RGB_CAMERA_RPY_RAD = tuple(math.radians(v) for v in TURTLEBOT_RGB_CAMERA_RPY_DEG)


# Empirical map -> splat correction found by perturbation scoring.
DEFAULT_SPLAT_MAP_X = 0.0
DEFAULT_SPLAT_MAP_Y = 0.0
DEFAULT_SPLAT_MAP_YAW = 0.0


def quaternion_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


TURTLEBOT_RGB_CAMERA_QUAT_XYZW = quaternion_xyzw_from_rpy(*TURTLEBOT_RGB_CAMERA_RPY_RAD)
