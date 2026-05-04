from __future__ import annotations

import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image as RosImage
from tf2_ros import Buffer, TransformException, TransformListener

from particle_filter.infrastructure.ros.image_conversion import ros_image_to_rgb_array
from particle_filter.infrastructure.ros.observation import (
    CameraIntrinsics,
    PoseMeasurement,
    RosTopicSettings,
    TurtleBotObservation,
)
from particle_filter.infrastructure.ros.tf_helpers import pose_measurement_from_transform, stamp_to_tuple


class TurtleBotObservationSource(Node):
    def __init__(self, settings: RosTopicSettings) -> None:
        super().__init__("turtlebot_observation_source")
        self._settings = settings
        self._image_message: RosImage | None = None
        self._camera_info_message: CameraInfo | None = None
        self._odometry_message: Odometry | None = None
        self._sequence_number = 0

        self.create_subscription(RosImage, settings.image_topic, self._handle_image, 10)
        self.create_subscription(CameraInfo, settings.camera_info_topic, self._handle_camera_info, 10)
        self.create_subscription(Odometry, settings.odometry_topic, self._handle_odometry, 20)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def is_running(self) -> bool:
        return rclpy.ok()

    def wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds

        while self.is_running() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._image_message is not None and self._camera_info_message is not None:
                if not self._settings.require_odometry or self._odometry_message is not None:
                    return

        missing_topics: list[str] = []
        if self._image_message is None:
            missing_topics.append(self._settings.image_topic)
        if self._camera_info_message is None:
            missing_topics.append(self._settings.camera_info_topic)
        if self._settings.require_odometry and self._odometry_message is None:
            missing_topics.append(self._settings.odometry_topic)

        raise TimeoutError(f"Timed out waiting for required ROS inputs: {', '.join(missing_topics)}")

    def spin_once(self, timeout_seconds: float) -> None:
        rclpy.spin_once(self, timeout_sec=timeout_seconds)

    def latest_image_stamp(self) -> tuple[int, int] | None:
        if self._image_message is None:
            return None
        return stamp_to_tuple(self._image_message.header.stamp)

    def read_latest_observation(self, *, include_map_pose: bool = True) -> TurtleBotObservation:
        if self._image_message is None or self._camera_info_message is None:
            raise RuntimeError("Observation source is not ready yet.")

        image_rgb = ros_image_to_rgb_array(self._image_message)
        camera_intrinsics = self._build_camera_intrinsics(self._camera_info_message)
        odometry_measurement = self._latest_odometry_measurement()
        map_measurement, resolved_tf_time, tf_error = self._lookup_map_pose() if include_map_pose else (None, None, None)
        image_stamp_seconds, image_stamp_nanoseconds = stamp_to_tuple(self._image_message.header.stamp)

        observation = TurtleBotObservation(
            sequence_number=self._sequence_number,
            image_rgb=image_rgb,
            image_encoding=self._image_message.encoding,
            image_frame_id=self._image_message.header.frame_id,
            image_stamp_seconds=image_stamp_seconds,
            image_stamp_nanoseconds=image_stamp_nanoseconds,
            camera=camera_intrinsics,
            odometry_pose=odometry_measurement.pose if odometry_measurement is not None else None,
            map_pose=map_measurement.pose if map_measurement is not None else None,
            resolved_tf_time=resolved_tf_time,
            tf_error=tf_error,
        )
        self._sequence_number += 1
        return observation

    def _handle_image(self, message: RosImage) -> None:
        self._image_message = message

    def _handle_camera_info(self, message: CameraInfo) -> None:
        self._camera_info_message = message

    def _handle_odometry(self, message: Odometry) -> None:
        self._odometry_message = message

    @staticmethod
    def _build_camera_intrinsics(message: CameraInfo) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=int(message.width),
            height=int(message.height),
            fx=float(message.k[0]),
            fy=float(message.k[4]),
            cx=float(message.k[2]),
            cy=float(message.k[5]),
            distortion_model=message.distortion_model,
            distortion_coefficients=tuple(float(value) for value in message.d),
        )

    def _latest_odometry_measurement(self) -> PoseMeasurement | None:
        if self._odometry_message is None:
            return None

        measurement = pose_measurement_from_transform(self._odometry_message.pose.pose)
        return PoseMeasurement(
            pose=measurement.pose,
            z=measurement.z,
            qx=measurement.qx,
            qy=measurement.qy,
            qz=measurement.qz,
            qw=measurement.qw,
            frame_id=self._odometry_message.header.frame_id,
            child_frame_id=self._odometry_message.child_frame_id,
            stamp_seconds=int(self._odometry_message.header.stamp.sec),
            stamp_nanoseconds=int(self._odometry_message.header.stamp.nanosec),
        )

    def _lookup_map_pose(self) -> tuple[PoseMeasurement | None, str | None, str | None]:
        if self._settings.tf_lookup_mode == "latest":
            return self._lookup_map_pose_at(Time(), "latest")

        try:
            image_time = Time.from_msg(self._image_message.header.stamp)
            return self._lookup_map_pose_at(image_time, "image")
        except TimeoutError as exc:
            if self._settings.tf_lookup_mode == "auto":
                self.get_logger().warn(
                    "Could not look up map->base at image timestamp; trying latest TF for debug pose. "
                    f"Original error: {exc}"
                )
                try:
                    return self._lookup_map_pose_at(Time(), "latest")
                except TimeoutError as latest_exc:
                    return None, None, str(latest_exc)
            return None, None, str(exc)

    def _lookup_map_pose_at(self, lookup_time: Time, resolved_tf_time: str) -> tuple[PoseMeasurement | None, str | None, str | None]:
        deadline = time.monotonic() + self._settings.tf_timeout_seconds
        last_error = None

        while self.is_running() and time.monotonic() < deadline:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._settings.map_frame,
                    self._settings.base_frame,
                    lookup_time,
                    timeout=Duration(seconds=0.1),
                )
                measurement = pose_measurement_from_transform(
                    transform,
                    child_frame_id=transform.child_frame_id,
                )
                return measurement, resolved_tf_time, None
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.02)

        raise TimeoutError(
            f"Timed out looking up TF {self._settings.map_frame} -> "
            f"{self._settings.base_frame}: {last_error}"
        )
