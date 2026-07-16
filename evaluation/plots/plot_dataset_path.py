#!/usr/bin/env python3
"""Plot a recorded replay dataset path on the occupancy-grid map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from evaluation.plots.plot_replay_paths import parse_map_yaml, world_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a recorded dataset trajectory on a PGM occupancy map.")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset directory containing manifest.json, or the manifest.json path itself.",
    )
    parser.add_argument("--map-yaml", type=Path, default=Path("cps_labor_map.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--show-waypoints", action="store_true")
    parser.add_argument("--full-map", action="store_true", help="Show the complete map instead of zooming to the path.")
    return parser.parse_args()


def resolve_manifest_path(dataset: Path) -> Path:
    if dataset.is_dir():
        return dataset / "manifest.json"
    return dataset


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def frame_pose(frame: dict) -> dict:
    pose = frame.get("pose") or frame.get("map_pose")
    if not pose:
        raise ValueError(f"Frame has no pose/map_pose: {frame}")
    return pose


def compute_bounds(points: list[tuple[float, float]], *, width: int, height: int, full_map: bool) -> tuple[float, float, float, float]:
    if full_map or not points:
        return 0.0, float(width), 0.0, float(height)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = 35.0
    return (
        max(0.0, min(xs) - pad),
        min(float(width), max(xs) + pad),
        max(0.0, min(ys) - pad),
        min(float(height), max(ys) + pad),
    )


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.dataset).resolve()
    manifest = load_manifest(manifest_path)
    frames = manifest.get("frames") or []
    if not frames:
        raise ValueError(f"Manifest contains no frames: {manifest_path}")

    map_metadata = parse_map_yaml(args.map_yaml.resolve())
    map_image = Image.open(map_metadata["image_path"]).convert("L")
    origin = list(map_metadata["origin"])
    resolution = float(map_metadata["resolution"])
    height = map_image.height

    poses = [frame_pose(frame) for frame in frames]
    path_points = [
        world_to_image(float(pose["x"]), float(pose["y"]), origin=origin, resolution=resolution, height=height)
        for pose in poses
    ]
    waypoint_points = []
    for goal in manifest.get("nav_goals") or []:
        waypoint_points.append(
            world_to_image(float(goal["x"]), float(goal["y"]), origin=origin, resolution=resolution, height=height)
        )

    x_min, x_max, y_min, y_max = compute_bounds(
        path_points + waypoint_points,
        width=map_image.width,
        height=map_image.height,
        full_map=args.full_map,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_image, cmap="gray", origin="upper")
    xs, ys = zip(*path_points)
    ax.plot(xs, ys, color="#ffb43c", linewidth=2.0, label="recorded path", zorder=3)
    ax.scatter([xs[0]], [ys[0]], color="#50dca0", s=45, marker="o", label="start", zorder=4)
    ax.scatter([xs[-1]], [ys[-1]], color="#ff4fd8", s=70, marker="*", label="end", zorder=4)

    if args.show_waypoints and waypoint_points:
        wxs, wys = zip(*waypoint_points)
        ax.scatter(wxs, wys, color="#4fc3ff", s=35, marker="x", label="waypoints", zorder=5)
        for index, (x_img, y_img) in enumerate(waypoint_points, start=1):
            ax.text(x_img + 4, y_img - 4, str(index), color="#4fc3ff", fontsize=9, weight="bold", zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(args.title or f"{manifest_path.parent.name} ({len(frames)} frames)")
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        output_path = manifest_path.parent / "dataset_path_overlay.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
