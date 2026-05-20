"""Observation value objects assembled from ROS topics and TF lookups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.particle_filter.domain.pose import Pose2D, wrap_angle


@dataclass(frozen=True)
class RosTopicSettings:
    image_topic: str = "/oakd/rgb/preview/image_raw"
    camera_info_topic: str = "/oakd/rgb/preview/camera_info"
    odometry_topic: str = "/odom"
    amcl_pose_topic: str = "/amcl_pose"
    map_frame: str = "map"
    base_frame: str = "base_link"
    tf_lookup_mode: str = "auto"
    tf_timeout_seconds: float = 0.5
    require_odometry: bool = True


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]


@dataclass(frozen=True)
class PoseMeasurement:
    pose: Pose2D
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    frame_id: str
    child_frame_id: str | None
    stamp_seconds: int
    stamp_nanoseconds: int


@dataclass(frozen=True)
class TurtleBotObservation:
    sequence_number: int
    image_rgb: np.ndarray
    image_encoding: str
    image_frame_id: str
    image_stamp_seconds: int
    image_stamp_nanoseconds: int
    camera: CameraIntrinsics
    odometry_pose: Pose2D | None
    map_pose: Pose2D | None
    amcl_pose: Pose2D | None
    resolved_tf_time: str | None
    tf_error: str | None

    def pose_error_against(self, estimated_pose: Pose2D) -> Pose2D:
        if self.map_pose is None:
            raise ValueError("Cannot compute pose error without a map pose.")
        return Pose2D(
            x=estimated_pose.x - self.map_pose.x,
            y=estimated_pose.y - self.map_pose.y,
            yaw=wrap_angle(estimated_pose.yaw - self.map_pose.yaw),
        )
