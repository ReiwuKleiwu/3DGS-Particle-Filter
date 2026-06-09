#!/usr/bin/env python3
"""Interactively align a top-down splat point projection over a ROS occupancy map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from PIL import Image

from core.config import load_turtlebot_localization_config


@dataclass(frozen=True)
class PlyHeader:
    vertex_count: int
    property_names: tuple[str, ...]
    data_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively align splat centers over a PGM map.")
    parser.add_argument("--ply", type=Path, default=Path("splat.ply"))
    parser.add_argument("--map-yaml", type=Path, default=Path("cps_labor_map.yaml"))
    parser.add_argument("--config", type=Path, default=Path("turtlebot_localization.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/artifacts/splat_map_alignment.json"))
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--x-range", type=float, default=3.0)
    parser.add_argument("--y-range", type=float, default=3.0)
    parser.add_argument("--scale-min", type=float, default=0.50)
    parser.add_argument("--scale-max", type=float, default=1.50)
    parser.add_argument("--yaw-range-deg", type=float, default=30.0)
    return parser.parse_args()


def parse_map_yaml(path: Path) -> dict:
    values: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "image":
            values[key] = value
        elif key == "resolution":
            values[key] = float(value)
        elif key == "origin":
            values[key] = [float(part.strip()) for part in value.strip("[]").split(",")]
    if "image" not in values or "resolution" not in values or "origin" not in values:
        raise ValueError(f"Map YAML is missing image/resolution/origin fields: {path}")
    image_path = Path(str(values["image"]))
    if not image_path.is_absolute():
        image_path = path.parent / image_path
    values["image_path"] = image_path.resolve()
    return values


def world_to_image(x: float, y: float, *, origin: list[float], resolution: float, height: int) -> tuple[float, float]:
    origin_x, origin_y, origin_yaw = origin
    dx = x - origin_x
    dy = y - origin_y
    cosine = math.cos(origin_yaw)
    sine = math.sin(origin_yaw)
    map_x = (cosine * dx + sine * dy) / resolution
    map_y = (-sine * dx + cosine * dy) / resolution
    return map_x, height - map_y


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
        rng = np.random.default_rng(42)
        indices = rng.choice(len(xy), size=max_points, replace=False)
        xy = xy[indices]
    return xy


def splat_xy_to_image_xy(
    splat_xy: np.ndarray,
    *,
    align_x: float,
    align_y: float,
    scale_x: float,
    scale_y: float,
    yaw_degrees: float,
    origin: list[float],
    resolution: float,
    height: int,
) -> np.ndarray:
    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("scale_x and scale_y must not be zero")
    yaw = math.radians(yaw_degrees)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation_inverse = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    rotated = (rotation_inverse @ (splat_xy - np.array([align_x, align_y], dtype=np.float64)).T).T
    map_xy = rotated / np.array([scale_x, scale_y], dtype=np.float64)
    image_xy = np.array(
        [
            world_to_image(float(x), float(y), origin=origin, resolution=resolution, height=height)
            for x, y in map_xy
        ],
        dtype=np.float64,
    )
    return image_xy


def main() -> None:
    args = parse_args()
    settings = load_turtlebot_localization_config(args.config)
    renderer = settings.renderer
    initial = {
        "x": float(renderer.splat_map_x),
        "y": float(renderer.splat_map_y),
        "scale_x": float(renderer.splat_map_scale),
        "scale_y": float(renderer.splat_map_scale),
        "yaw": float(renderer.splat_map_yaw_degrees),
    }

    map_metadata = parse_map_yaml(args.map_yaml.resolve())
    map_image = Image.open(map_metadata["image_path"])
    origin = map_metadata["origin"]
    resolution = map_metadata["resolution"]
    splat_xy = read_splat_xy(args.ply.resolve(), max_points=args.max_points)

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.34)
    ax.imshow(map_image, cmap="gray", origin="upper")
    points = splat_xy_to_image_xy(
        splat_xy,
        align_x=initial["x"],
        align_y=initial["y"],
        scale_x=initial["scale_x"],
        scale_y=initial["scale_y"],
        yaw_degrees=initial["yaw"],
        origin=origin,
        resolution=resolution,
        height=map_image.height,
    )
    scatter = ax.scatter(points[:, 0], points[:, 1], s=args.point_size, alpha=args.alpha, color="#e8590c", linewidths=0)
    ax.set_xlim(0, map_image.width)
    ax.set_ylim(map_image.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    title = ax.set_title("", fontsize=10)

    slider_axes = {
        "x": fig.add_axes([0.12, 0.24, 0.70, 0.025]),
        "y": fig.add_axes([0.12, 0.20, 0.70, 0.025]),
        "scale_x": fig.add_axes([0.12, 0.16, 0.70, 0.025]),
        "scale_y": fig.add_axes([0.12, 0.12, 0.70, 0.025]),
        "yaw": fig.add_axes([0.12, 0.08, 0.70, 0.025]),
    }
    sliders = {
        "x": Slider(slider_axes["x"], "x", initial["x"] - args.x_range, initial["x"] + args.x_range, valinit=initial["x"]),
        "y": Slider(slider_axes["y"], "y", initial["y"] - args.y_range, initial["y"] + args.y_range, valinit=initial["y"]),
        "scale_x": Slider(slider_axes["scale_x"], "scale x", args.scale_min, args.scale_max, valinit=initial["scale_x"]),
        "scale_y": Slider(slider_axes["scale_y"], "scale y", args.scale_min, args.scale_max, valinit=initial["scale_y"]),
        "yaw": Slider(
            slider_axes["yaw"],
            "yaw deg",
            initial["yaw"] - args.yaw_range_deg,
            initial["yaw"] + args.yaw_range_deg,
            valinit=initial["yaw"],
        ),
    }

    def current_values() -> dict[str, float]:
        return {
            "splat_map_x": float(sliders["x"].val),
            "splat_map_y": float(sliders["y"].val),
            "splat_map_scale": float((sliders["scale_x"].val + sliders["scale_y"].val) * 0.5),
            "splat_map_scale_x": float(sliders["scale_x"].val),
            "splat_map_scale_y": float(sliders["scale_y"].val),
            "splat_map_yaw_degrees": float(sliders["yaw"].val),
        }

    def update(_=None) -> None:
        values = current_values()
        next_points = splat_xy_to_image_xy(
            splat_xy,
            align_x=values["splat_map_x"],
            align_y=values["splat_map_y"],
            scale_x=values["splat_map_scale_x"],
            scale_y=values["splat_map_scale_y"],
            yaw_degrees=values["splat_map_yaw_degrees"],
            origin=origin,
            resolution=resolution,
            height=map_image.height,
        )
        scatter.set_offsets(next_points)
        title.set_text(
            "renderer:\n"
            f"  splat_map_x: {values['splat_map_x']:.6f}    "
            f"splat_map_y: {values['splat_map_y']:.6f}    "
            f"scale_x: {values['splat_map_scale_x']:.6f}    "
            f"scale_y: {values['splat_map_scale_y']:.6f}    "
            f"splat_map_yaw_degrees: {values['splat_map_yaw_degrees']:.6f}"
        )
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    save_axis = fig.add_axes([0.84, 0.24, 0.13, 0.05])
    print_axis = fig.add_axes([0.84, 0.17, 0.13, 0.05])
    reset_axis = fig.add_axes([0.84, 0.10, 0.13, 0.05])
    save_button = Button(save_axis, "Save")
    print_button = Button(print_axis, "Print")
    reset_button = Button(reset_axis, "Reset")

    def print_values(_=None) -> None:
        values = current_values()
        print("renderer:")
        for key, value in values.items():
            print(f"  {key}: {value:.6f}")

    def save_values(_=None) -> None:
        values = current_values()
        payload = {"renderer": values}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print_values()
        print(f"Wrote {args.output}")

    def reset_values(_=None) -> None:
        sliders["x"].set_val(initial["x"])
        sliders["y"].set_val(initial["y"])
        sliders["scale_x"].set_val(initial["scale_x"])
        sliders["scale_y"].set_val(initial["scale_y"])
        sliders["yaw"].set_val(initial["yaw"])

    save_button.on_clicked(save_values)
    print_button.on_clicked(print_values)
    reset_button.on_clicked(reset_values)
    update()
    plt.show()


if __name__ == "__main__":
    main()
