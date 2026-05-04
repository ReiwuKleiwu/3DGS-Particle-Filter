from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization
from plyfile import PlyData
from config import (
    DEFAULT_SPLAT_MAP_X,
    DEFAULT_SPLAT_MAP_Y,
    DEFAULT_SPLAT_MAP_YAW,
    TURTLEBOT_RGB_CAMERA_QUAT_XYZW,
    TURTLEBOT_RGB_CAMERA_XYZ,
)


@dataclass
class CameraSpec:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_3dgs_ply(path: Path, device: torch.device, sh_layout: str, opacity_threshold: float):
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    data = ply["vertex"].data
    names = data.dtype.names or ()

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PLY is missing required 3DGS properties: {missing}")

    means_np = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    scales_np = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32))
    quats_np = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    opacities_np = sigmoid(np.asarray(data["opacity"], dtype=np.float32))

    dc_names = sorted([name for name in names if name.startswith("f_dc_")], key=lambda n: int(n.rsplit("_", 1)[1]))
    rest_names = sorted([name for name in names if name.startswith("f_rest_")], key=lambda n: int(n.rsplit("_", 1)[1]))
    if len(dc_names) != 3:
        raise ValueError(f"Expected 3 f_dc coefficients, found {len(dc_names)}")

    sh0_np = np.stack([data[name] for name in dc_names], axis=1).astype(np.float32)
    if rest_names:
        rest_np = np.stack([data[name] for name in rest_names], axis=1).astype(np.float32)
        rest_bases = len(rest_names) // 3
        if len(rest_names) % 3 != 0:
            raise ValueError(f"f_rest count must be divisible by 3, found {len(rest_names)}")
        if sh_layout == "official":
            shn_np = rest_np.reshape(-1, 3, rest_bases).transpose(0, 2, 1)
        elif sh_layout == "basis-major":
            shn_np = rest_np.reshape(-1, rest_bases, 3)
        else:
            raise ValueError(f"Unknown SH layout: {sh_layout}")
        colors_np = np.concatenate([sh0_np[:, None, :], shn_np], axis=1)
        sh_degree = int(math.sqrt(colors_np.shape[1]) - 1)
    else:
        colors_np = sh0_np[:, None, :]
        sh_degree = 0

    if opacity_threshold > 0.0:
        keep = opacities_np >= opacity_threshold
        means_np = means_np[keep]
        scales_np = scales_np[keep]
        quats_np = quats_np[keep]
        opacities_np = opacities_np[keep]
        colors_np = colors_np[keep]

    means = torch.from_numpy(np.asarray(means_np)).to(device)
    scales = torch.from_numpy(np.asarray(scales_np)).to(device)
    quats = torch.nn.functional.normalize(torch.from_numpy(np.asarray(quats_np)).to(device), dim=-1)
    opacities = torch.from_numpy(np.asarray(opacities_np)).to(device)
    colors = torch.from_numpy(np.asarray(colors_np)).to(device)
    return means, quats, scales, opacities, colors, sh_degree


def quaternion_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm")
    qx, qy, qz, qw = q / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_from_xyz_quat(
    xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
    device: torch.device,
) -> torch.Tensor:
    qx, qy, qz, qw = quat_xyzw
    rot = quaternion_xyzw_to_rotation_matrix(qx, qy, qz, qw)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot
    transform[:3, 3] = np.array(xyz, dtype=np.float32)
    return torch.from_numpy(transform).to(device)


def pose2d_to_matrix(x: float, y: float, yaw: float, device: torch.device) -> torch.Tensor:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transform = torch.eye(4, dtype=torch.float32, device=device)
    transform[0, 0] = cos_yaw
    transform[0, 1] = -sin_yaw
    transform[1, 0] = sin_yaw
    transform[1, 1] = cos_yaw
    transform[0, 3] = x
    transform[1, 3] = y
    return transform


class SplatRenderer:
    def __init__(
        self,
        ply_path: str | Path | None = None,
        sh_layout: str = "official",
        opacity_threshold: float = 0.0,
        default_packed: bool = False,
        default_radius_clip: float = 3.0,
        default_max_batch_size: int = 20,
        base_camera_xyz: tuple[float, float, float] = TURTLEBOT_RGB_CAMERA_XYZ,
        base_camera_quat_xyzw: tuple[float, float, float, float] = TURTLEBOT_RGB_CAMERA_QUAT_XYZW,
        splat_map_xy_yaw: tuple[float, float, float] = (DEFAULT_SPLAT_MAP_X, DEFAULT_SPLAT_MAP_Y, DEFAULT_SPLAT_MAP_YAW),
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the renderer service")

        self.device = torch.device("cuda")
        self.ply_path = Path(ply_path or os.environ.get("SPLAT_PATH", "/workspace/splat.ply"))
        self.sh_layout = sh_layout
        self.default_packed = default_packed
        self.default_radius_clip = default_radius_clip
        self.default_max_batch_size = default_max_batch_size
        self.base_camera_xyz = base_camera_xyz
        self.base_camera_quat_xyzw = base_camera_quat_xyzw
        self.splat_map_xy_yaw = splat_map_xy_yaw

        self.means, self.quats, self.scales, self.opacities, self.colors, self.max_sh_degree = load_3dgs_ply(
            self.ply_path,
            self.device,
            self.sh_layout,
            opacity_threshold,
        )

    def _make_K(self, camera: CameraSpec) -> torch.Tensor:
        return torch.tensor(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

    def _viewmats_from_poses(self, poses: list[Pose2D]) -> torch.Tensor:
        viewmats = []
        splat_map_x, splat_map_y, splat_map_yaw = self.splat_map_xy_yaw
        splat_t_map = pose2d_to_matrix(splat_map_x, splat_map_y, splat_map_yaw, self.device)
        base_t_camera = transform_from_xyz_quat(self.base_camera_xyz, self.base_camera_quat_xyzw, self.device)

        for pose in poses:
            world_t_base = pose2d_to_matrix(pose.x, pose.y, pose.yaw, self.device)
            world_t_camera = world_t_base @ base_t_camera
            splat_t_camera = splat_t_map @ world_t_camera
            viewmats.append(torch.linalg.inv(splat_t_camera))
        return torch.stack(viewmats, dim=0)

    def _background(self, count: int, packed: bool, white_background: bool) -> torch.Tensor:
        background = (
            torch.ones((3,), dtype=torch.float32, device=self.device)
            if white_background
            else torch.zeros((3,), dtype=torch.float32, device=self.device)
        )
        return background if packed else background.expand(count, 3).contiguous()

    def _active_colors(self, sh_degree: int | None) -> tuple[torch.Tensor, int]:
        active_degree = self.max_sh_degree if sh_degree is None else sh_degree
        if active_degree > self.max_sh_degree:
            raise ValueError(f"Requested SH degree {active_degree}, but file only has degree {self.max_sh_degree}")
        colors = self.colors[:, : (active_degree + 1) ** 2, :].contiguous()
        return colors, active_degree

    @torch.inference_mode()
    def render_batch(
        self,
        poses: list[Pose2D],
        camera: CameraSpec,
        packed: bool | None = None,
        white_background: bool = False,
        radius_clip: float | None = None,
        sh_degree: int | None = None,
    ) -> torch.Tensor:
        if not poses:
            raise ValueError("At least one pose is required")
        packed = self.default_packed if packed is None else packed
        radius_clip = self.default_radius_clip if radius_clip is None else radius_clip
        colors, active_sh_degree = self._active_colors(sh_degree)
        K = self._make_K(camera)
        viewmats = self._viewmats_from_poses(poses)
        backgrounds = self._background(len(poses), packed, white_background)

        render_colors, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            colors,
            viewmats,
            K[None].expand(len(poses), 3, 3).contiguous(),
            camera.width,
            camera.height,
            sh_degree=active_sh_degree,
            packed=packed,
            render_mode="RGB",
            radius_clip=radius_clip,
            backgrounds=backgrounds,
        )
        return render_colors[..., :3]

    @torch.inference_mode()
    def render_batch_chunked(
        self,
        poses: list[Pose2D],
        camera: CameraSpec,
        packed: bool | None = None,
        white_background: bool = False,
        radius_clip: float | None = None,
        sh_degree: int | None = None,
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        if not poses:
            raise ValueError("At least one pose is required")

        chunk_size = max_batch_size or self.default_max_batch_size
        if chunk_size <= 0:
            raise ValueError("max_batch_size must be positive")

        rendered_chunks = []
        for start_index in range(0, len(poses), chunk_size):
            pose_chunk = poses[start_index:start_index + chunk_size]
            rendered_chunks.append(
                self.render_batch(
                    poses=pose_chunk,
                    camera=camera,
                    packed=packed,
                    white_background=white_background,
                    radius_clip=radius_clip,
                    sh_degree=sh_degree,
                )
            )
            torch.cuda.empty_cache()

        return torch.cat(rendered_chunks, dim=0)
