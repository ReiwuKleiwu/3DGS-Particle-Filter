#!/usr/bin/env python3
"""Plot a top-down splat point overlay on an occupancy-grid map."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.rendering.config import DEFAULT_SPLAT_MAP_X, DEFAULT_SPLAT_MAP_Y, DEFAULT_SPLAT_MAP_YAW
from evaluation.plots.plot_replay_paths import parse_map_yaml, world_to_image


@dataclass(frozen=True)
class PlyHeader:
    vertex_count: int
    property_names: tuple[str, ...]
    data_offset: int


def parse_binary_ply_header(path: Path) -> PlyHeader:
    property_names: list[str] = []
    vertex_count: int | None = None
    data_offset = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            data_offset += len(line)
            stripped = line.decode("ascii").strip()
            if stripped in {"ply", "format binary_little_endian 1.0"} or stripped.startswith("comment "):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot splat Gaussian centers over a ROS occupancy map.")
    parser.add_argument("--ply", type=Path, default=Path("splat.ply"))
    parser.add_argument("--map-yaml", type=Path, default=Path("cps_labor_map.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/artifacts/splat_map_overlay.png"))
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--full-map", action="store_true")
    return parser.parse_args()


def read_splat_xy(path: Path, *, max_points: int) -> np.ndarray:
    header = parse_binary_ply_header(path)
    dtype = make_structured_dtype(header.property_names)
    if "x" not in header.property_names or "y" not in header.property_names:
        raise ValueError(f"PLY must contain x and y properties: {path}")

    with path.open("rb") as handle:
        handle.seek(header.data_offset)
        data = np.fromfile(handle, dtype=dtype, count=header.vertex_count)

    xy = np.column_stack([data["x"], data["y"]]).astype(np.float64, copy=False)
    if max_points > 0 and len(xy) > max_points:
        indices = np.linspace(0, len(xy) - 1, max_points, dtype=np.int64)
        xy = xy[indices]
    return xy


def splat_xy_to_map_xy(splat_xy: np.ndarray, *, x: float, y: float, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation_inverse = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    translated = splat_xy - np.array([x, y], dtype=np.float64)
    return (rotation_inverse @ translated.T).T


def main() -> None:
    args = parse_args()
    map_metadata = parse_map_yaml(args.map_yaml.resolve())
    map_image = Image.open(map_metadata["image_path"])
    origin = map_metadata["origin"]
    resolution = map_metadata["resolution"]
    height = map_image.height

    splat_xy = read_splat_xy(args.ply.resolve(), max_points=args.max_points)
    map_xy = splat_xy_to_map_xy(
        splat_xy,
        x=DEFAULT_SPLAT_MAP_X,
        y=DEFAULT_SPLAT_MAP_Y,
        yaw=DEFAULT_SPLAT_MAP_YAW,
    )
    image_xy = np.array(
        [
            world_to_image(float(x), float(y), origin=origin, resolution=resolution, height=height)
            for x, y in map_xy
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(image_xy).all(axis=1)
    image_xy = image_xy[finite]
    inside = (
        (image_xy[:, 0] >= 0)
        & (image_xy[:, 0] < map_image.width)
        & (image_xy[:, 1] >= 0)
        & (image_xy[:, 1] < map_image.height)
    )
    image_xy = image_xy[inside]

    fig, ax = plt.subplots(figsize=(9.0, 9.0), constrained_layout=True)
    ax.imshow(map_image, cmap="gray", origin="upper")
    ax.scatter(
        image_xy[:, 0],
        image_xy[:, 1],
        s=args.point_size,
        alpha=args.alpha,
        color="#d9480f",
        linewidths=0,
    )

    if args.full_map or len(image_xy) == 0:
        ax.set_xlim(0, map_image.width)
        ax.set_ylim(map_image.height, 0)
    else:
        pad = 80
        ax.set_xlim(max(0, float(image_xy[:, 0].min()) - pad), min(map_image.width, float(image_xy[:, 0].max()) + pad))
        ax.set_ylim(min(map_image.height, float(image_xy[:, 1].max()) + pad), max(0, float(image_xy[:, 1].min()) - pad))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Splat centers over occupancy map\n"
        f"renderer config.py: x={DEFAULT_SPLAT_MAP_X:.4f}, y={DEFAULT_SPLAT_MAP_Y:.4f}, "
        f"yaw={math.degrees(DEFAULT_SPLAT_MAP_YAW):.3f} deg",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} ({len(image_xy)} plotted points inside map)")


if __name__ == "__main__":
    main()
