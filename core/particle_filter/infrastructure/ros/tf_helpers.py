"""Helpers for converting TF and pose messages into backend pose types."""

from __future__ import annotations

import numpy as np

from core.particle_filter.domain.pose import Pose2D
from core.particle_filter.infrastructure.ros.observation import PoseMeasurement


def stamp_to_tuple(stamp) -> tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    sine_yaw = 2.0 * (w * z + x * y)
    cosine_yaw = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(sine_yaw, cosine_yaw))


def pose_measurement_from_transform(transform, *, child_frame_id: str | None = None) -> PoseMeasurement:
    if hasattr(transform, "translation") and hasattr(transform, "rotation"):
        translation = transform.translation
        rotation = transform.rotation
        header = getattr(transform, "header", None)
    elif hasattr(transform, "transform"):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        header = getattr(transform, "header", None)
    elif hasattr(transform, "position") and hasattr(transform, "orientation"):
        translation = transform.position
        rotation = transform.orientation
        header = getattr(transform, "header", None)
    else:
        raise TypeError(f"Unsupported transform/pose object: {type(transform)!r}")

    frame_id = header.frame_id if header is not None else ""
    stamp_seconds, stamp_nanoseconds = stamp_to_tuple(header.stamp) if header is not None else (0, 0)

    yaw = yaw_from_quaternion_xyzw(
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    )

    return PoseMeasurement(
        pose=Pose2D(
            x=float(translation.x),
            y=float(translation.y),
            yaw=yaw,
        ),
        z=float(translation.z),
        qx=float(rotation.x),
        qy=float(rotation.y),
        qz=float(rotation.z),
        qw=float(rotation.w),
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        stamp_seconds=stamp_seconds,
        stamp_nanoseconds=stamp_nanoseconds,
    )
