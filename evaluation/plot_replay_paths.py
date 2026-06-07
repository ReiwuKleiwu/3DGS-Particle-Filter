"""Plot replay ground-truth and particle-filter paths on the occupancy-grid map."""

from __future__ import annotations

import argparse
import csv
import math
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

from evaluation.plot_style import ERROR_LINK_COLOR, GT_COLOR, PF_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PF estimate paths against replay ground truth on a PGM map.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, default=Path("map.yaml"))
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--prior-case-index", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=["local", "global"])
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--path-id", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--show-error-links", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def sort_splat_ids(splat_ids: list[str]) -> list[str]:
    def key(splat_id: str) -> tuple[int, str]:
        digits = "".join(ch for ch in splat_id if ch.isdigit())
        return (int(digits) if digits else 0, splat_id)

    return sorted(splat_ids, key=key)


def group_rows(
    rows: list[dict],
    *,
    seed: int,
    prior_case_index: int | None,
    mode: str | None,
    scenario_id: str | None,
    path_id: str | None,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(float(row["seed"])) != seed:
            continue
        if prior_case_index is not None and int(float(row.get("prior_case_index", 0))) != prior_case_index:
            continue
        if mode is not None and row.get("localization_mode") != mode:
            continue
        if scenario_id is not None and row.get("scenario_id") != scenario_id:
            continue
        if path_id is not None and row.get("path_id") != path_id:
            continue
        grouped[row["splat_id"]].append(row)
    for splat_rows in grouped.values():
        splat_rows.sort(key=lambda row: int(float(row["frame_index"])))
    return grouped


def available_values(rows: list[dict], key: str) -> list[str]:
    return sorted({row.get(key, "") for row in rows if row.get(key, "")})


def axis_bounds(grouped: dict[str, list[dict]], *, origin: list[float], resolution: float, height: int) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for rows in grouped.values():
        for row in rows:
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
    pad = 35
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def plot_panel(
    ax,
    *,
    map_image,
    rows: list[dict],
    splat_id: str,
    origin: list[float],
    resolution: float,
    bounds: tuple[float, float, float, float],
    show_error_links: bool,
) -> None:
    height = map_image.height
    ax.imshow(map_image, cmap="gray", origin="upper")
    gt_points = [
        world_to_image(float(row["truth_x"]), float(row["truth_y"]), origin=origin, resolution=resolution, height=height)
        for row in rows
    ]
    est_points = [
        world_to_image(float(row["estimate_x"]), float(row["estimate_y"]), origin=origin, resolution=resolution, height=height)
        for row in rows
    ]
    gt_xs, gt_ys = zip(*gt_points)
    est_xs, est_ys = zip(*est_points)

    if show_error_links:
        step = max(1, len(rows) // 18)
        for gt, est in zip(gt_points[::step], est_points[::step]):
            error_line = ax.plot(
                [gt[0], est[0]],
                [gt[1], est[1]],
                color=ERROR_LINK_COLOR,
                alpha=0.55,
                linewidth=0.9,
                zorder=2,
            )[0]

    gt_line = ax.plot(gt_xs, gt_ys, color=GT_COLOR, linewidth=2.0, linestyle="-", label="GT", zorder=3)[0]
    pf_line = ax.plot(
        est_xs,
        est_ys,
        color=PF_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="PF estimate",
        zorder=4,
    )[0]
    ax.scatter([gt_xs[0]], [gt_ys[0]], color=GT_COLOR, s=18, marker="o", zorder=5)
    ax.scatter([gt_xs[-1]], [gt_ys[-1]], color=GT_COLOR, s=28, marker="*", zorder=5)
    ax.scatter([est_xs[-1]], [est_ys[-1]], color=PF_COLOR, s=24, marker="x", linewidths=1.1, zorder=5)

    mean_error = sum(float(row["translation_error_m"]) for row in rows) / len(rows)
    final_error = float(rows[-1]["translation_error_m"])
    ax.set_title(f"{splat_id}\nmean={mean_error:.3f}m final={final_error:.3f}m", fontsize=9)
    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    per_frame_path = input_dir / "per_frame.csv"
    if not per_frame_path.is_file():
        raise FileNotFoundError(f"Missing per-frame CSV: {per_frame_path}")

    map_metadata = parse_map_yaml(args.map_yaml.resolve())
    map_image = Image.open(map_metadata["image_path"]).convert("L")
    origin = list(map_metadata["origin"])
    resolution = float(map_metadata["resolution"])

    all_rows = read_csv(per_frame_path)
    grouped = group_rows(
        all_rows,
        seed=args.seed,
        prior_case_index=args.prior_case_index,
        mode=args.mode,
        scenario_id=args.scenario_id,
        path_id=args.path_id,
    )
    if not grouped:
        raise ValueError(
            "No rows found for the requested filters. "
            f"Available scenario_id={available_values(all_rows, 'scenario_id')}, "
            f"path_id={available_values(all_rows, 'path_id')}, "
            f"seed={available_values(all_rows, 'seed')}, "
            f"localization_mode={available_values(all_rows, 'localization_mode')}."
        )

    splat_ids = sort_splat_ids(list(grouped))
    bounds = axis_bounds(grouped, origin=origin, resolution=resolution, height=map_image.height)

    columns = min(3, len(splat_ids))
    rows_count = math.ceil(len(splat_ids) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(4.2 * columns, 3.6 * rows_count + 0.7))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, splat_id in zip(axes_list, splat_ids):
        plot_panel(
            ax,
            map_image=map_image,
            rows=grouped[splat_id],
            splat_id=splat_id,
            origin=origin,
            resolution=resolution,
            bounds=bounds,
            show_error_links=args.show_error_links,
        )
    for ax in axes_list[len(splat_ids):]:
        ax.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    title_parts = [f"seed={args.seed}"]
    if args.mode:
        title_parts.append(f"mode={args.mode}")
    if args.scenario_id:
        title_parts.append(f"scenario={args.scenario_id}")
    if args.path_id:
        title_parts.append(f"path={args.path_id}")
    if args.prior_case_index is not None:
        title_parts.append(f"prior_case={args.prior_case_index}")
    fig.suptitle(f"Replay Path Overlay: {', '.join(title_parts)}", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.11, wspace=0.05, hspace=0.24)

    output_base = input_dir / (
        args.output_name or f"path_overlay_seed{args.seed}"
    )
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_base.with_suffix('.png')}")
    print(f"Wrote {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
