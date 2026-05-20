#!/usr/bin/env python3
"""Records ROS observations and poses into replay datasets for offline evaluation."""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image as RosImage
from tf2_ros import Buffer, TransformException, TransformListener

from core.replay_tuning.paths import DATASETS_DIR


def ros_image_to_numpy(msg: RosImage) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    if encoding in {"rgb8", "bgr8"}:
        image = data.reshape(height, msg.step)[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return image.copy()
    if encoding in {"rgba8", "bgra8"}:
        image = data.reshape(height, msg.step)[:, : width * 4].reshape(height, width, 4)
        if encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3].copy()
    if encoding in {"mono8", "8uc1"}:
        image = data.reshape(height, msg.step)[:, :width]
        return np.repeat(image[:, :, None], 3, axis=2).copy()

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def stamp_to_dict(stamp) -> dict:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class ReplayRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("record_replay_dataset")
        self.args = args
        self.image_msg = None
        self.camera_info_msg = None
        self.odom_msg = None
        self.cmd_vel_msg = None
        self.last_recorded_image_stamp = None
        self.sequence = 0
        self.frames = []
        self.cmd_vel_history = []
        self.nav_feedback_history = []
        self.nav_goal_sent_at = None
        self.nav_result = None

        self.create_subscription(RosImage, args.image_topic, self._image_callback, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_callback, 10)
        self.create_subscription(Odometry, args.odom_topic, self._odom_callback, 20)
        self.create_subscription(Twist, args.cmd_vel_topic, self._cmd_vel_callback, 50)

        self.nav_client = ActionClient(self, NavigateToPose, args.navigate_to_pose_action)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.run_dir = args.out_dir / args.name
        self.images_dir = self.run_dir / "images"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _image_callback(self, msg: RosImage) -> None:
        self.image_msg = msg

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info_msg = msg

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_msg = msg

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.cmd_vel_msg = msg
        self.cmd_vel_history.append(
            {
                "stamp_sec": self.get_clock().now().nanoseconds / 1e9,
                "linear_x": float(msg.linear.x),
                "linear_y": float(msg.linear.y),
                "linear_z": float(msg.linear.z),
                "angular_x": float(msg.angular.x),
                "angular_y": float(msg.angular.y),
                "angular_z": float(msg.angular.z),
            }
        )

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.image_msg is not None and self.camera_info_msg is not None and self.odom_msg is not None:
                return
        missing = []
        if self.image_msg is None:
            missing.append(self.args.image_topic)
        if self.camera_info_msg is None:
            missing.append(self.args.camera_info_topic)
        if self.odom_msg is None:
            missing.append(self.args.odom_topic)
        raise TimeoutError(f"Timed out waiting for: {', '.join(missing)}")

    def current_odom_pose(self) -> dict | None:
        if self.odom_msg is None:
            return None
        pose = self.odom_msg.pose.pose
        q = pose.orientation
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "qx": float(q.x),
            "qy": float(q.y),
            "qz": float(q.z),
            "qw": float(q.w),
            "yaw": yaw_from_quaternion_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
            "frame_id": self.odom_msg.header.frame_id,
            "child_frame_id": self.odom_msg.child_frame_id,
            "stamp": stamp_to_dict(self.odom_msg.header.stamp),
        }

    def lookup_map_pose(self) -> tuple[dict, dict]:
        if self.args.tf_time == "latest":
            return self._lookup_map_pose_at(Time())
        try:
            return self._lookup_map_pose_at(Time.from_msg(self.image_msg.header.stamp))
        except TimeoutError as exc:
            if self.args.tf_time != "auto":
                raise
            self.get_logger().warn(
                "Could not look up map->base at image timestamp; falling back to latest TF. "
                f"Original error: {exc}"
            )
            return self._lookup_map_pose_at(Time())

    def _lookup_map_pose_at(self, lookup_time: Time) -> tuple[dict, dict]:
        deadline = time.monotonic() + self.args.tf_timeout
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    self.args.base_frame,
                    lookup_time,
                    timeout=Duration(seconds=0.1),
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                return (
                    {
                        "x": float(t.x),
                        "y": float(t.y),
                        "z": float(t.z),
                        "qx": float(q.x),
                        "qy": float(q.y),
                        "qz": float(q.z),
                        "qw": float(q.w),
                        "yaw": yaw_from_quaternion_xyzw(float(q.x), float(q.y), float(q.z), float(q.w)),
                        "stamp": stamp_to_dict(transform.header.stamp),
                    },
                    {
                        "resolved_tf_time": "latest" if lookup_time.nanoseconds == 0 else "image",
                        "tf_stamp": stamp_to_dict(transform.header.stamp),
                    },
                )
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.02)
        raise TimeoutError(f"Timed out looking up TF {self.args.map_frame} -> {self.args.base_frame}: {last_error}")

    @staticmethod
    def atomic_save_image(image: np.ndarray, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        Image.fromarray(image, mode="RGB").save(tmp_path, format="PNG")
        os.replace(tmp_path, path)

    def record_frame(self) -> bool:
        if self.image_msg is None:
            return False
        stamp_key = (self.image_msg.header.stamp.sec, self.image_msg.header.stamp.nanosec)
        if stamp_key == self.last_recorded_image_stamp:
            return False

        map_pose, tf_metadata = self.lookup_map_pose()
        odom_pose = self.current_odom_pose()
        image_name = f"frame_{self.sequence:06d}.png"
        image_path = self.images_dir / image_name
        image = ros_image_to_numpy(self.image_msg)
        self.atomic_save_image(image, image_path)

        self.frames.append(
            {
                "image_path": str(Path("images") / image_name),
                "pose": {
                    "x": map_pose["x"],
                    "y": map_pose["y"],
                    "yaw": map_pose["yaw"],
                },
                "map_pose": map_pose,
                "odom_pose": odom_pose,
                "image_stamp": stamp_to_dict(self.image_msg.header.stamp),
                "tf_metadata": tf_metadata,
            }
        )
        self.last_recorded_image_stamp = stamp_key
        self.sequence += 1
        self.get_logger().info(
            f"recorded frame {self.sequence:04d} "
            f"x={map_pose['x']:.3f}, y={map_pose['y']:.3f}, yaw={math.degrees(map_pose['yaw']):.1f}deg"
        )
        return True

    def wait_and_record_frame(self, timeout_sec: float | None = None) -> bool:
        deadline = time.monotonic() + (timeout_sec if timeout_sec is not None else self.args.timeout)
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                if self.record_frame():
                    return True
            except TimeoutError as exc:
                last_error = exc
                self.get_logger().warn(
                    "Skipping frame because pose lookup was not ready yet. "
                    f"Waiting for a fresher image/TF pair. Error: {exc}"
                )
            rclpy.spin_once(self, timeout_sec=0.05)
        if last_error is not None:
            raise TimeoutError(f"Timed out waiting for a recordable frame: {last_error}")
        return False

    def _handle_nav_feedback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose
        current_orientation = current_pose.orientation
        self.nav_feedback_history.append(
            {
                "stamp_sec": self.get_clock().now().nanoseconds / 1e9,
                "current_pose": {
                    "x": float(current_pose.position.x),
                    "y": float(current_pose.position.y),
                    "z": float(current_pose.position.z),
                    "yaw": yaw_from_quaternion_xyzw(
                        float(current_orientation.x),
                        float(current_orientation.y),
                        float(current_orientation.z),
                        float(current_orientation.w),
                    ),
                },
                "navigation_time_sec": float(feedback.navigation_time.sec) + float(feedback.navigation_time.nanosec) / 1e9,
                "estimated_time_remaining_sec": float(feedback.estimated_time_remaining.sec)
                + float(feedback.estimated_time_remaining.nanosec) / 1e9,
                "number_of_recoveries": int(feedback.number_of_recoveries),
                "distance_remaining": float(feedback.distance_remaining),
            }
        )

    def send_nav_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=self.args.timeout):
            raise TimeoutError(f"Timed out waiting for Nav2 action server {self.args.navigate_to_pose_action}")

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.args.map_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(self.args.goal_x)
        goal_pose.pose.position.y = float(self.args.goal_y)
        goal_pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(float(self.args.goal_yaw))
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self._handle_nav_feedback)
        deadline = time.monotonic() + self.args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if send_future.done():
                break
        if not send_future.done():
            raise TimeoutError("Timed out waiting for Nav2 goal acceptance")

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Nav2 goal was rejected")
        self.nav_goal_sent_at = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info(
            f"sent nav2 goal x={self.args.goal_x:.3f}, y={self.args.goal_y:.3f}, "
            f"yaw={math.degrees(self.args.goal_yaw):.1f}deg"
        )
        return goal_handle.get_result_async()

    def run(self) -> None:
        self.wait_until_ready()
        self.get_logger().info("Recorder is ready")

        settle_until = time.monotonic() + self.args.settle_time
        while rclpy.ok() and time.monotonic() < settle_until:
            rclpy.spin_once(self, timeout_sec=0.05)

        next_record_time = time.monotonic()
        self.wait_and_record_frame()

        result_future = self.send_nav_goal()
        deadline = time.monotonic() + self.args.goal_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_record_time:
                self.wait_and_record_frame(timeout_sec=0.25)
                next_record_time = now + 1.0 / self.args.record_rate_hz
            if result_future.done():
                result = result_future.result()
                status = result.status
                self.nav_result = {
                    "status": int(status),
                    "status_name": self.goal_status_name(status),
                    "finished_at_sec": self.get_clock().now().nanoseconds / 1e9,
                }
                if status != GoalStatus.STATUS_SUCCEEDED:
                    raise RuntimeError(f"Nav2 goal finished with status {status}")
                self.get_logger().info("Nav2 goal reached")
                break
            rclpy.spin_once(self, timeout_sec=0.02)
        else:
            raise TimeoutError("Timed out waiting for Nav2 goal result")

        final_deadline = time.monotonic() + self.args.final_settle_time
        while rclpy.ok() and time.monotonic() < final_deadline:
            now = time.monotonic()
            if now >= next_record_time:
                self.wait_and_record_frame(timeout_sec=0.25)
                next_record_time = now + 1.0 / self.args.record_rate_hz
            rclpy.spin_once(self, timeout_sec=0.02)

    def write_outputs(self) -> tuple[Path, Path]:
        if not self.frames:
            raise RuntimeError("No frames were recorded")

        camera_info = self.camera_info_msg
        manifest_path = self.run_dir / "manifest.json"
        poses_csv_path = self.run_dir / "poses.csv"
        raw_json_path = self.run_dir / "raw_capture.json"

        first_pose = self.frames[0]["pose"]
        prior = {
            "x": first_pose["x"] + self.args.prior_offset_x,
            "y": first_pose["y"] + self.args.prior_offset_y,
            "yaw": first_pose["yaw"] + math.radians(self.args.prior_offset_yaw_deg),
            "sigma_x": self.args.prior_sigma_x,
            "sigma_y": self.args.prior_sigma_y,
            "sigma_yaw_deg": self.args.prior_sigma_yaw_deg,
        }

        manifest = {
            "ply_path": str(self.args.ply.resolve()),
            "camera_info": {
                "width": int(camera_info.width),
                "height": int(camera_info.height),
                "k": [float(v) for v in camera_info.k],
            },
            "initial_prior": prior,
            "frames": [
                {
                    "image_path": frame["image_path"],
                    "pose": frame["pose"],
                    "odom_pose": frame["odom_pose"],
                    "image_stamp": frame["image_stamp"],
                    "tf_metadata": frame["tf_metadata"],
                }
                for frame in self.frames
            ],
            "nav_goal": {
                "x": self.args.goal_x,
                "y": self.args.goal_y,
                "yaw": self.args.goal_yaw,
                "sent_at_sec": self.nav_goal_sent_at,
                "result": self.nav_result,
            },
            "notes": self.args.notes,
        }

        with poses_csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "x", "y", "z", "qx", "qy", "qz", "qw"])
            for frame in self.frames:
                map_pose = frame["map_pose"]
                writer.writerow(
                    [
                        frame["image_path"],
                        map_pose["x"],
                        map_pose["y"],
                        map_pose["z"],
                        map_pose["qx"],
                        map_pose["qy"],
                        map_pose["qz"],
                        map_pose["qw"],
                    ]
                )

        manifest_path.write_text(json.dumps(manifest, indent=2))
        raw_json_path.write_text(
            json.dumps(
                {
                    "image_topic": self.args.image_topic,
                    "camera_info_topic": self.args.camera_info_topic,
                    "odom_topic": self.args.odom_topic,
                    "map_frame": self.args.map_frame,
                    "base_frame": self.args.base_frame,
                    "navigate_to_pose_action": self.args.navigate_to_pose_action,
                    "nav_goal": {
                        "x": self.args.goal_x,
                        "y": self.args.goal_y,
                        "yaw": self.args.goal_yaw,
                        "sent_at_sec": self.nav_goal_sent_at,
                        "result": self.nav_result,
                    },
                    "cmd_vel_history": self.cmd_vel_history,
                    "nav_feedback_history": self.nav_feedback_history,
                    "frames": self.frames,
                    "camera_info": {
                        "width": int(camera_info.width),
                        "height": int(camera_info.height),
                        "distortion_model": camera_info.distortion_model,
                        "d": [float(v) for v in camera_info.d],
                        "k": [float(v) for v in camera_info.k],
                        "r": [float(v) for v in camera_info.r],
                        "p": [float(v) for v in camera_info.p],
                    },
                },
                indent=2,
            )
        )
        return manifest_path, poses_csv_path

    @staticmethod
    def goal_status_name(status: int) -> str:
        names = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return names.get(status, f"STATUS_{status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a Nav2 NavigateToPose goal and record a replay-tuning dataset."
    )
    parser.add_argument("--image-topic", default="/oakd/rgb/preview/image_raw")
    parser.add_argument("--camera-info-topic", default="/oakd/rgb/preview/camera_info")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--navigate-to-pose-action", default="/navigate_to_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tf-time", choices=["auto", "image", "latest"], default="auto")
    parser.add_argument("--tf-timeout", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--name", required=True, help="Subdirectory name under out-dir.")
    parser.add_argument("--ply", type=Path, default=Path("splat.ply"))
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--goal-yaw", type=float, required=True, help="Goal yaw in radians.")
    parser.add_argument("--goal-timeout", type=float, default=180.0)
    parser.add_argument("--record-rate-hz", type=float, default=2.0)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--final-settle-time", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--prior-offset-x", type=float, default=0.0)
    parser.add_argument("--prior-offset-y", type=float, default=0.0)
    parser.add_argument("--prior-offset-yaw-deg", type=float, default=0.0)
    parser.add_argument("--prior-sigma-x", type=float, default=0.5)
    parser.add_argument("--prior-sigma-y", type=float, default=0.5)
    parser.add_argument("--prior-sigma-yaw-deg", type=float, default=30.0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ReplayRecorder(args)
    try:
        node.run()
        manifest_path, poses_csv_path = node.write_outputs()
        print(f"Wrote manifest: {manifest_path}")
        print(f"Wrote poses CSV: {poses_csv_path}")
        print(f"Recorded {len(node.frames)} frames under {node.run_dir}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
