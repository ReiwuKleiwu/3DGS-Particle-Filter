"""Plot replay ground-truth and particle-filter paths on the occupancy-grid map."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from evaluation.plots.plot_style import ERROR_LINK_COLOR, GT_COLOR, PF_COLOR, TELEPORT_COLOR


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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Plot every scenario/path/mode/prior/seed/splat combination separately.")
    parser.add_argument("--show-error-links", action="store_true")
    parser.add_argument(
        "--teleport-threshold-m",
        type=float,
        default=1.0,
        help="Break and mark reference-path jumps larger than this distance. Use 0 to disable.",
    )
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


def safe_name(*parts: object) -> str:
    tokens = []
    for part in parts:
        if part is None:
            continue
        value = str(part).strip()
        if not value:
            continue
        tokens.append(
            value.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
    return "__".join(tokens)


def display_name(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    text = text.replace("small_house_", "")
    text = text.replace("_small_house_", "_")
    text = text.replace("_small_house", "")
    text = text.replace("small_house", "")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def plot_output_dir(base_dir: Path, *, mode: str, prior_case_index: str, seed: str) -> Path:
    mode_dir = safe_name(mode or "unknown_mode")
    seed_dir = safe_name(f"seed_{int(float(seed))}" if seed else "seed_unknown")
    parts = [base_dir, mode_dir]
    if mode == "local":
        prior_label = f"prior_{int(float(prior_case_index))}" if prior_case_index else "prior_unknown"
        parts.append(safe_name(prior_label))
    parts.append(seed_dir)
    return Path(*parts)


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


def group_all_rows(rows: list[dict]) -> dict[tuple[str, str, str, str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("scenario_id", ""),
            row.get("path_id", ""),
            row.get("localization_mode", ""),
            row.get("prior_case_index", ""),
            row.get("seed", ""),
            row.get("splat_id", ""),
        )
        grouped[key].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: int(float(row["frame_index"])))
    return grouped


def available_values(rows: list[dict], key: str) -> list[str]:
    return sorted({row.get(key, "") for row in rows if row.get(key, "")})


def rows_for_scenario_scope(rows: list[dict], scenario_id: str | None) -> list[dict]:
    if scenario_id is None:
        return rows
    return [row for row in rows if row.get("scenario_id") == scenario_id]


def should_show_path_label(rows: list[dict], *, scenario_id: str | None) -> bool:
    path_ids = {row.get("path_id", "") for row in rows_for_scenario_scope(rows, scenario_id) if row.get("path_id", "")}
    return len(path_ids) > 1


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


def draw_direction_chevrons(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str,
    zorder: int,
) -> None:
    if len(points) < 2:
        return
    step = max(1, len(points) // 8)
    arrow_indices = list(range(step, len(points), step))
    if arrow_indices and arrow_indices[-1] == len(points) - 1:
        arrow_indices = arrow_indices[:-1]
    chevron_length = 5.0
    chevron_width = 3.2
    for index in arrow_indices:
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            continue
        ux = dx / distance
        uy = dy / distance
        px = -uy
        py = ux
        left = (
            x1 - ux * chevron_length + px * chevron_width,
            y1 - uy * chevron_length + py * chevron_width,
        )
        right = (
            x1 - ux * chevron_length - px * chevron_width,
            y1 - uy * chevron_length - py * chevron_width,
        )
        ax.plot(
            [left[0], x1, right[0]],
            [left[1], y1, right[1]],
            color=color,
            linewidth=1.25,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )


def split_reference_segments(
    rows: list[dict],
    points: list[tuple[float, float]],
    *,
    teleport_threshold_m: float,
) -> tuple[list[list[tuple[float, float]]], list[tuple[tuple[float, float], tuple[float, float], float]]]:
    if not points:
        return [], []
    if teleport_threshold_m <= 0:
        return [points], []

    segments: list[list[tuple[float, float]]] = [[points[0]]]
    teleports: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for index in range(1, len(points)):
        previous_row = rows[index - 1]
        current_row = rows[index]
        distance_m = math.hypot(
            float(current_row["truth_x"]) - float(previous_row["truth_x"]),
            float(current_row["truth_y"]) - float(previous_row["truth_y"]),
        )
        if distance_m > teleport_threshold_m:
            teleports.append((points[index - 1], points[index], distance_m))
            segments.append([points[index]])
        else:
            segments[-1].append(points[index])
    return segments, teleports


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
    teleport_threshold_m: float,
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
    gt_segments, teleports = split_reference_segments(
        rows,
        gt_points,
        teleport_threshold_m=teleport_threshold_m,
    )

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

    for segment_index, segment in enumerate(gt_segments):
        if len(segment) < 2:
            continue
        segment_xs, segment_ys = zip(*segment)
        ax.plot(
            segment_xs,
            segment_ys,
            color=GT_COLOR,
            linewidth=2.0,
            linestyle="-",
            label="Referenz" if segment_index == 0 else None,
            zorder=3,
        )
    for teleport_index, (start, end, distance_m) in enumerate(teleports):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=TELEPORT_COLOR,
            linewidth=2.2,
            linestyle=(0, (2.0, 2.0)),
            alpha=0.95,
            label="Kidnapped-Robot-Ereignis" if teleport_index == 0 else None,
            zorder=7,
        )
        ax.scatter([start[0]], [start[1]], color=TELEPORT_COLOR, s=34, marker="x", linewidths=1.2, zorder=8)
        ax.scatter(
            [end[0]],
            [end[1]],
            facecolors="none",
            edgecolors=TELEPORT_COLOR,
            s=42,
            marker="o",
            linewidths=1.3,
            zorder=8,
        )
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        ax.text(
            mid_x,
            mid_y,
            f"{distance_m:.1f} m",
            color=TELEPORT_COLOR,
            fontsize=7,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=9,
        )
    pf_line = ax.plot(
        est_xs,
        est_ys,
        color=PF_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="PF-Schätzung",
        zorder=4,
    )[0]
    for segment in gt_segments:
        draw_direction_chevrons(ax, segment, color=GT_COLOR, zorder=5)
    draw_direction_chevrons(ax, est_points, color=PF_COLOR, zorder=6)
    ax.scatter([gt_xs[0]], [gt_ys[0]], color=GT_COLOR, s=18, marker="o", zorder=5)
    ax.scatter([gt_xs[-1]], [gt_ys[-1]], color=GT_COLOR, s=28, marker="*", zorder=5)
    ax.scatter([est_xs[-1]], [est_ys[-1]], color=PF_COLOR, s=24, marker="x", linewidths=1.1, zorder=5)

    mean_error = sum(float(row["translation_error_m"]) for row in rows) / len(rows)
    final_error = float(rows[-1]["translation_error_m"])
    ax.set_title(f"{display_name(splat_id)}\nMittelwert={mean_error:.3f}m Final={final_error:.3f}m", fontsize=9)
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
    if args.all:
        plot_groups = group_all_rows(all_rows)
        output_dir = args.output_dir or input_dir / (args.output_name or "path_overlays_all")
    else:
        grouped = group_rows(
            all_rows,
            seed=args.seed,
            prior_case_index=args.prior_case_index,
            mode=args.mode,
            scenario_id=args.scenario_id,
            path_id=args.path_id,
        )
        plot_groups = {
            (
                args.scenario_id or "",
                args.path_id or "",
                args.mode or "",
                str(args.prior_case_index) if args.prior_case_index is not None else "",
                str(args.seed),
                splat_id,
            ): rows
            for splat_id, rows in grouped.items()
        }
        output_dir = args.output_dir or input_dir / (args.output_name or f"path_overlay_seed{args.seed}")

    if not plot_groups:
        raise ValueError(
            "No rows found for the requested filters. "
            f"Available scenario_id={available_values(all_rows, 'scenario_id')}, "
            f"path_id={available_values(all_rows, 'path_id')}, "
            f"seed={available_values(all_rows, 'seed')}, "
            f"localization_mode={available_values(all_rows, 'localization_mode')}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for scenario_id, path_id, mode, prior_case_index, seed, splat_id in sorted(
        plot_groups,
        key=lambda key: (
            key[0],
            key[1],
            key[2],
            int(float(key[3])) if key[3] else -1,
            int(float(key[4])) if key[4] else -1,
            sort_splat_ids([key[5]])[0],
        ),
    ):
        rows = plot_groups[(scenario_id, path_id, mode, prior_case_index, seed, splat_id)]
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        bounds = axis_bounds({splat_id: rows}, origin=origin, resolution=resolution, height=map_image.height)
        plot_panel(
            ax,
            map_image=map_image,
            rows=rows,
            splat_id=splat_id,
            origin=origin,
            resolution=resolution,
            bounds=bounds,
            show_error_links=args.show_error_links,
            teleport_threshold_m=args.teleport_threshold_m,
        )

        handles, labels = ax.get_legend_handles_labels()
        title_parts = [f"Seed={seed}"]
        if mode:
            mode_label = "lokal" if mode == "local" else "global" if mode == "global" else mode
            title_parts.append(f"Modus={mode_label}")
        if scenario_id:
            title_parts.append(f"Szenario={display_name(scenario_id)}")
        if path_id and should_show_path_label(all_rows, scenario_id=scenario_id):
            title_parts.append(f"Pfad={display_name(path_id)}")
        if prior_case_index:
            title_parts.append(f"Prior-Fall={prior_case_index}")
        fig.suptitle(f"Pfadvergleich: {', '.join(title_parts)}", y=0.98, fontsize=8)
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
        fig.subplots_adjust(left=0.03, right=0.99, top=0.89, bottom=0.14)

        per_plot_output_dir = plot_output_dir(
            output_dir,
            mode=mode,
            prior_case_index=prior_case_index,
            seed=seed,
        )
        per_plot_output_dir.mkdir(parents=True, exist_ok=True)
        output_base = per_plot_output_dir / safe_name(
            "path_overlay",
            scenario_id,
            path_id,
            mode,
            f"prior{prior_case_index}" if prior_case_index else "",
            f"seed{seed}",
            splat_id,
        )
        png_path = output_base.with_suffix(".png")
        pdf_path = output_base.with_suffix(".pdf")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        written_paths.extend([png_path, pdf_path])

    print(f"Wrote {len(written_paths)} files to {output_dir}")


if __name__ == "__main__":
    main()
