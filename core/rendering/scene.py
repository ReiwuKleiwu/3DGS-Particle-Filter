"""Shared scene loading, transform math, and PLY conversion helpers for render backends."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.rendering.types import CameraSpec, Pose2D
from core.rendering.config import (
    DEFAULT_SPLAT_MAP_X,
    DEFAULT_SPLAT_MAP_Y,
    DEFAULT_SPLAT_MAP_YAW,
    TURTLEBOT_RGB_CAMERA_QUAT_XYZW,
    TURTLEBOT_RGB_CAMERA_XYZ,
)


EXPECTED_PLY_PROPERTIES = (
    ["x", "y", "z"]
    + ["nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


@dataclass(frozen=True)
class PlyHeader:
    vertex_count: int
    property_names: tuple[str, ...]
    data_offset: int


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
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quaternion_xyzw_to_rotation_matrix(*quat_xyzw)
    transform[:3, 3] = np.array(xyz, dtype=np.float32)
    return transform


def pose2d_to_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = cos_yaw
    transform[0, 1] = -sin_yaw
    transform[1, 0] = sin_yaw
    transform[1, 1] = cos_yaw
    transform[0, 3] = x
    transform[1, 3] = y
    return transform


def camera_world_transform(pose: Pose2D) -> np.ndarray:
    splat_t_map = pose2d_to_matrix(DEFAULT_SPLAT_MAP_X, DEFAULT_SPLAT_MAP_Y, DEFAULT_SPLAT_MAP_YAW)
    world_t_base = pose2d_to_matrix(pose.x, pose.y, pose.yaw)
    base_t_camera = transform_from_xyz_quat(TURTLEBOT_RGB_CAMERA_XYZ, TURTLEBOT_RGB_CAMERA_QUAT_XYZW)
    return splat_t_map @ world_t_base @ base_t_camera


def camera_entry(pose: Pose2D, camera: CameraSpec, img_name: str) -> dict:
    world_t_camera = camera_world_transform(pose)
    return {
        "id": img_name,
        "img_name": img_name,
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "position": world_t_camera[:3, 3].tolist(),
        "rotation": world_t_camera[:3, :3].tolist(),
    }


def write_cameras_json(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def parse_binary_ply_header(path: Path) -> PlyHeader:
    property_names: list[str] = []
    vertex_count: int | None = None
    data_offset = 0

    with path.open("rb") as fin:
        while True:
            line = fin.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            data_offset += len(line)
            stripped = line.decode("ascii").strip()
            if stripped == "ply":
                continue
            if stripped == "format binary_little_endian 1.0":
                continue
            if stripped.startswith("comment "):
                continue
            if stripped.startswith("element "):
                parts = stripped.split()
                if len(parts) == 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                continue
            if stripped.startswith("property "):
                parts = stripped.split()
                if len(parts) != 3 or parts[1] != "float":
                    raise ValueError(f"Unsupported PLY property in {path}: {stripped}")
                property_names.append(parts[2])
                continue
            if stripped == "end_header":
                break
            raise ValueError(f"Unsupported PLY header line in {path}: {stripped}")

    if vertex_count is None:
        raise ValueError(f"PLY header missing vertex count: {path}")
    return PlyHeader(vertex_count=vertex_count, property_names=tuple(property_names), data_offset=data_offset)


def make_structured_dtype(property_names: tuple[str, ...]) -> np.dtype:
    return np.dtype([(name, "<f4") for name in property_names], align=False)


def write_vkdiff_header(fout, vertex_count: int) -> None:
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
        *[f"property float {name}" for name in EXPECTED_PLY_PROPERTIES],
        "end_header",
    ]
    fout.write(("\n".join(header_lines) + "\n").encode("ascii"))


def convert_point_cloud_to_vkdiff(source_ply: Path, target_ply: Path, chunk_size: int = 65536) -> int:
    header = parse_binary_ply_header(source_ply)
    source_dtype = make_structured_dtype(header.property_names)
    source_names = set(header.property_names)
    required = {"x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - source_names)
    if missing:
        raise ValueError(f"Source PLY is missing required properties: {', '.join(missing)}")

    target_ply.parent.mkdir(parents=True, exist_ok=True)
    with source_ply.open("rb") as fin, target_ply.open("wb") as fout:
        fin.seek(header.data_offset)
        write_vkdiff_header(fout, header.vertex_count)

        remaining = header.vertex_count
        column_map = {name: idx for idx, name in enumerate(EXPECTED_PLY_PROPERTIES)}
        while remaining > 0:
            count = min(chunk_size, remaining)
            chunk = np.fromfile(fin, dtype=source_dtype, count=count)
            if len(chunk) != count:
                raise ValueError(
                    f"Unexpected EOF while reading vertex data from {source_ply}: expected {count}, got {len(chunk)}"
                )

            out = np.zeros((count, len(EXPECTED_PLY_PROPERTIES)), dtype="<f4")
            for axis in ("x", "y", "z", "opacity"):
                out[:, column_map[axis]] = chunk[axis]
            for name in ("scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
                out[:, column_map[name]] = chunk[name]
            for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
                if name in source_names:
                    out[:, column_map[name]] = chunk[name]
            for rest_idx in range(45):
                name = f"f_rest_{rest_idx}"
                if name in source_names:
                    out[:, column_map[name]] = chunk[name]
            for name in ("nx", "ny", "nz"):
                if name in source_names:
                    out[:, column_map[name]] = chunk[name]

            fout.write(out.tobytes(order="C"))
            remaining -= count
    return header.vertex_count
