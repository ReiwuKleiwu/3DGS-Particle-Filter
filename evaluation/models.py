"""Replay manifest, candidate, and summary models used by offline tuning tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.particle_filter.domain.pose import Pose2D, Pose2DPrior, wrap_angle
from core.particle_filter.infrastructure.ros.observation import CameraIntrinsics


@dataclass(frozen=True)
class ReplayFrame:
    image_path: str
    pose: Pose2D
    odom_pose: Pose2D | None
    image_stamp_seconds: int
    image_stamp_nanoseconds: int
    resolved_tf_time: str | None
    tf_error: str | None


@dataclass(frozen=True)
class ReplayManifest:
    ply_path: str
    camera: CameraIntrinsics
    initial_prior: Pose2DPrior
    frames: list[ReplayFrame]
    notes: str = ""
    manifest_path: Path | None = None

    @staticmethod
    def load(path: Path) -> "ReplayManifest":
        """Loads a replay manifest JSON file into strongly typed replay models."""
        payload = json.loads(path.read_text())
        camera_payload = payload["camera_info"]
        camera = CameraIntrinsics(
            width=int(camera_payload["width"]),
            height=int(camera_payload["height"]),
            fx=float(camera_payload["k"][0]),
            fy=float(camera_payload["k"][4]),
            cx=float(camera_payload["k"][2]),
            cy=float(camera_payload["k"][5]),
            distortion_model="plumb_bob",
            distortion_coefficients=tuple(),
        )
        prior_payload = payload["initial_prior"]
        initial_prior = Pose2DPrior(
            mean=Pose2D(
                x=float(prior_payload["x"]),
                y=float(prior_payload["y"]),
                yaw=float(prior_payload["yaw"]),
            ),
            sigma_x=float(prior_payload["sigma_x"]),
            sigma_y=float(prior_payload["sigma_y"]),
            sigma_yaw=np.deg2rad(float(prior_payload["sigma_yaw_deg"])),
        )
        frames = []
        for frame_payload in payload["frames"]:
            odom_payload = frame_payload.get("odom_pose")
            tf_metadata = frame_payload.get("tf_metadata") or {}
            image_stamp = frame_payload.get("image_stamp") or {"sec": 0, "nanosec": 0}
            frames.append(
                ReplayFrame(
                    image_path=frame_payload["image_path"],
                    pose=Pose2D(
                        x=float(frame_payload["pose"]["x"]),
                        y=float(frame_payload["pose"]["y"]),
                        yaw=float(frame_payload["pose"]["yaw"]),
                    ),
                    odom_pose=(
                        None
                        if odom_payload is None
                        else Pose2D(
                            x=float(odom_payload["x"]),
                            y=float(odom_payload["y"]),
                            yaw=float(odom_payload["yaw"]),
                        )
                    ),
                    image_stamp_seconds=int(image_stamp["sec"]),
                    image_stamp_nanoseconds=int(image_stamp["nanosec"]),
                    resolved_tf_time=tf_metadata.get("resolved_tf_time"),
                    tf_error=tf_metadata.get("tf_error"),
                )
            )
        return ReplayManifest(
            ply_path=str(payload["ply_path"]),
            camera=camera,
            initial_prior=initial_prior,
            frames=frames,
            notes=str(payload.get("notes", "")),
            manifest_path=path,
        )

    def resolve_image_path(self, image_path: str) -> Path:
        """Resolves a manifest image reference relative to the manifest location when needed."""
        path = Path(image_path)
        if path.is_absolute():
            return path
        base_dir = self.manifest_path.parent if self.manifest_path is not None else Path.cwd()
        return (base_dir / path).resolve()


@dataclass(frozen=True)
class PriorOffset:
    dx: float
    dy: float
    dyaw_degrees: float

    def apply(self, base_pose: Pose2D, sigma_x: float, sigma_y: float, sigma_yaw_degrees: float) -> Pose2DPrior:
        """Builds a perturbed prior around a ground-truth pose for robustness testing."""
        return Pose2DPrior(
            mean=Pose2D(
                x=base_pose.x + self.dx,
                y=base_pose.y + self.dy,
                yaw=wrap_angle(base_pose.yaw + np.deg2rad(self.dyaw_degrees)),
            ),
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_yaw=np.deg2rad(sigma_yaw_degrees),
        )


DEFAULT_PRIOR_BANK = [
    PriorOffset(dx=0.5, dy=0.0, dyaw_degrees=20.0),
    PriorOffset(dx=-0.5, dy=0.0, dyaw_degrees=-20.0),
    PriorOffset(dx=0.0, dy=0.75, dyaw_degrees=45.0),
    PriorOffset(dx=0.0, dy=-0.75, dyaw_degrees=-45.0),
    PriorOffset(dx=1.0, dy=-0.5, dyaw_degrees=90.0),
]


@dataclass(frozen=True)
class SearchCandidate:
    particle_count: int
    resample_threshold_ratio: float
    temperature: float
    motion_noise_x: float
    motion_noise_y: float
    motion_noise_yaw: float
    hybrid_ssim_weight: float
    hybrid_l1_weight: float
    hybrid_gradient_weight: float
    lpips_top_k: int
    lpips_weight: float
    random_seed: int
    prior_sigma_x: float
    prior_sigma_y: float
    prior_sigma_yaw_degrees: float


@dataclass(frozen=True)
class TrialSummary:
    mean_translation_error_m: float
    median_translation_error_m: float
    max_translation_error_m: float
    mean_abs_yaw_error_degrees: float
    catastrophic_failure_rate: float
    mean_elapsed_ms: float
    objective: float


@dataclass(frozen=True)
class TrialResult:
    candidate: SearchCandidate
    summary: TrialSummary
    case_results: list[dict]
