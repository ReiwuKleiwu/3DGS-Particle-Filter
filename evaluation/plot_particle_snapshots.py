"""Render recorded particle snapshots as map-overlay PNG frames."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from evaluation.plot_replay_paths import parse_map_yaml, read_csv, world_to_image
from evaluation.plot_style import GT_COLOR, PARTICLE_COLOR, PF_ALT_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recorded PF particle snapshots on the occupancy-grid map.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, default=Path("map.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--every", type=int, default=1, help="Render every Nth recorded frame.")
    return parser.parse_args()


def group_particles(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(float(row["frame_index"]))].append(row)
    return dict(sorted(grouped.items()))


def frame_rows_by_index(rows: list[dict], *, run_id: str) -> dict[int, dict]:
    return {
        int(float(row["frame_index"])): row
        for row in rows
        if row.get("run_id") == run_id
    }


def axis_bounds(particle_rows: list[dict], per_frame_rows: dict[int, dict], *, origin, resolution, height):
    xs = []
    ys = []
    for row in particle_rows:
        x_img, y_img = world_to_image(float(row["x"]), float(row["y"]), origin=origin, resolution=resolution, height=height)
        xs.append(x_img)
        ys.append(y_img)
    for row in per_frame_rows.values():
        for prefix in ("truth", "estimate"):
            x_img, y_img = world_to_image(
                float(row[f"{prefix}_x"]),
                float(row[f"{prefix}_y"]),
                origin=origin,
                resolution=resolution,
                height=height,
            )
            xs.append(x_img)
            ys.append(y_img)
    if not xs or not ys:
        return 0, 1, 0, 1
    pad = 45
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be positive")

    input_dir = args.input_dir.resolve()
    particle_csv = input_dir / "particles" / f"{args.run_id}.csv"
    if not particle_csv.is_file():
        raise FileNotFoundError(f"Missing particle snapshot CSV: {particle_csv}")
    per_frame_csv = input_dir / "per_frame.csv"
    if not per_frame_csv.is_file():
        raise FileNotFoundError(f"Missing per-frame CSV: {per_frame_csv}")

    map_metadata = parse_map_yaml(args.map_yaml.resolve())
    map_image = Image.open(map_metadata["image_path"])
    origin = map_metadata["origin"]
    resolution = map_metadata["resolution"]
    height = map_image.height

    particle_rows = read_csv(particle_csv)
    particles_by_frame = group_particles(particle_rows)
    per_frame_rows = frame_rows_by_index(read_csv(per_frame_csv), run_id=args.run_id)
    bounds = axis_bounds(particle_rows, per_frame_rows, origin=origin, resolution=resolution, height=height)

    output_dir = args.output_dir or (input_dir / "particle_plots" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_count = 0
    for sequence_index, (frame_index, rows) in enumerate(particles_by_frame.items()):
        if sequence_index % args.every != 0:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
        ax.imshow(map_image, cmap="gray", origin="upper")

        xs = []
        ys = []
        for row in rows:
            x_img, y_img = world_to_image(float(row["x"]), float(row["y"]), origin=origin, resolution=resolution, height=height)
            xs.append(x_img)
            ys.append(y_img)
        ax.scatter(
            xs,
            ys,
            color=PARTICLE_COLOR,
            s=5,
            alpha=0.45,
            linewidths=0,
        )

        frame_row = per_frame_rows.get(frame_index)
        if frame_row is not None:
            truth = world_to_image(
                float(frame_row["truth_x"]),
                float(frame_row["truth_y"]),
                origin=origin,
                resolution=resolution,
                height=height,
            )
            estimate = world_to_image(
                float(frame_row["estimate_x"]),
                float(frame_row["estimate_y"]),
                origin=origin,
                resolution=resolution,
                height=height,
            )
            ax.scatter(
                [truth[0]],
                [truth[1]],
                marker="x",
                s=38,
                color=GT_COLOR,
                linewidths=1.2,
                label="GT",
            )
            ax.scatter(
                [estimate[0]],
                [estimate[1]],
                marker="o",
                s=28,
                color=PF_ALT_COLOR,
                label="PF",
            )
            ax.legend(frameon=False, fontsize=8, loc="upper right")

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[3], bounds[2])
        ax.set_aspect("equal")
        ax.set_title(f"{args.run_id}\nframe={frame_index}", fontsize=9)
        ax.axis("off")
        fig.savefig(output_dir / f"frame_{frame_index:06d}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        rendered_count += 1

    print(f"Wrote {rendered_count} particle snapshot frames to {output_dir}")


if __name__ == "__main__":
    main()
