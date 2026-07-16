#!/usr/bin/env python3
"""Aggregate and plot matrix experiment outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plots.plot_style import COLORBLIND_SAFE_COLORS, LINE_STYLES, MARKERS


SUMMARY_KEYS = [
    "mean_translation_error_m",
    "median_translation_error_m",
    "p95_translation_error_m",
    "mean_yaw_error_degrees",
    "p95_yaw_error_degrees",
    "mean_combined_pose_error_m",
    "p95_combined_pose_error_m",
    "failure_rate",
    "lost_tracking_rate",
    "converged",
    "mean_best_metric_value",
    "median_best_metric_value",
    "p95_best_metric_value",
    "final_best_metric_value",
    "mean_median_metric_value",
    "p95_median_metric_value",
    "mean_render_and_score_ms",
    "mean_total_frame_ms",
    "mean_total_hz",
    "mean_gpu_memory_used_mb",
    "max_gpu_memory_used_mb",
    "p95_gpu_memory_used_mb",
    "mean_x_bias_m",
    "mean_y_bias_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze outputs from evaluation/experiments/run_matrix_experiment.py.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--splat-iterations",
        type=int,
        nargs="+",
        default=None,
        help="Only include these splat training iterations in plots and summaries.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def filter_by_splat_iterations(rows: list[dict], iterations: set[int] | None) -> list[dict]:
    if not iterations:
        return rows
    return [
        row
        for row in rows
        if row.get("training_iteration", "") != ""
        and int(float(row["training_iteration"])) in iterations
    ]


def summarize_by_condition(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_id"],
            row["localization_mode"],
            row.get("prior_case_index", ""),
            row.get("prior_dx", ""),
            row.get("prior_dy", ""),
            row.get("prior_dyaw_degrees", ""),
            row["splat_id"],
            row.get("training_iteration", ""),
            row.get("particle_count", ""),
            row.get("metric_name", ""),
        )
        grouped[key].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        scenario_id, mode, prior_case_index, prior_dx, prior_dy, prior_dyaw_degrees, splat_id, training_iteration, particle_count, metric_name = key
        row = {
            "scenario_id": scenario_id,
            "localization_mode": mode,
            "prior_case_index": prior_case_index,
            "prior_dx": prior_dx,
            "prior_dy": prior_dy,
            "prior_dyaw_degrees": prior_dyaw_degrees,
            "splat_id": splat_id,
            "training_iteration": training_iteration,
            "particle_count": particle_count,
            "metric_name": metric_name,
            "run_count": len(group_rows),
            "path_count": len({item["path_id"] for item in group_rows}),
            "seed_count": len({item["seed"] for item in group_rows}),
            "renderer_gaussians": group_rows[0].get("renderer_gaussians", ""),
            "splat_file_size_mb": group_rows[0].get("splat_file_size_mb", ""),
        }
        for metric in SUMMARY_KEYS:
            values = [to_float(item, metric) for item in group_rows]
            row[f"mean_{metric}"] = mean(values)
            row[f"median_{metric}"] = median(values)
            row[f"std_{metric}"] = stdev(values)
        summaries.append(row)

    summaries.sort(
        key=lambda row: (
            row["scenario_id"],
            row["localization_mode"],
            int(float(row["prior_case_index"])) if row["prior_case_index"] else -1,
            int(float(row["training_iteration"])) if row["training_iteration"] else -1,
        )
    )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def plot_metric_by_splat(
    rows: list[dict],
    *,
    metric_key: str,
    ylabel: str,
    output_path: Path,
    std_key: str | None = None,
) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], row["localization_mode"], row.get("prior_case_index", ""))].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for series_index, ((scenario_id, mode, prior_case_index), group_rows) in enumerate(sorted(grouped.items())):
        ordered = sorted(group_rows, key=lambda row: int(float(row["training_iteration"])))
        xs = [int(float(row["training_iteration"])) for row in ordered]
        ys = [float(row[metric_key]) for row in ordered]
        mode_label = "lokal" if mode == "local" else "global" if mode == "global" else mode
        prior_label = f" / Prior {prior_case_index}" if mode == "local" and prior_case_index != "" else ""
        color = COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[series_index % len(MARKERS)],
            linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
            color=color,
            linewidth=1.7,
            label=f"{display_name(scenario_id)} / {mode_label}{prior_label}",
        )
        if std_key is not None and all(row.get(std_key, "") != "" for row in ordered):
            stds = [float(row[std_key]) for row in ordered]
            lower = [y - std for y, std in zip(ys, stds)]
            upper = [y + std for y, std in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)

    ax.set_xlabel("Splat-Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_particle_count(
    rows: list[dict],
    *,
    metric_key: str,
    ylabel: str,
    output_path: Path,
    std_key: str | None = None,
) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["scenario_id"],
                row["localization_mode"],
                row.get("prior_case_index", ""),
                row.get("splat_id", ""),
            )
        ].append(row)

    series = {
        key: group_rows
        for key, group_rows in grouped.items()
        if len({row.get("particle_count", "") for row in group_rows}) > 1
    }
    if not series:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for series_index, ((scenario_id, mode, prior_case_index, splat_id), group_rows) in enumerate(sorted(series.items())):
        ordered = sorted(group_rows, key=lambda row: int(float(row["particle_count"])))
        xs = [int(float(row["particle_count"])) for row in ordered]
        ys = [float(row[metric_key]) for row in ordered]
        mode_label = "lokal" if mode == "local" else "global" if mode == "global" else mode
        prior_label = f" / Prior {prior_case_index}" if mode == "local" and prior_case_index != "" else ""
        iteration = ordered[0].get("training_iteration", "")
        splat_label = f"{int(float(iteration))}" if iteration else display_name(splat_id)
        color = COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[series_index % len(MARKERS)],
            linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
            color=color,
            linewidth=1.7,
            label=f"{display_name(scenario_id)} / {mode_label}{prior_label} / {splat_label}",
        )
        if std_key is not None and all(row.get(std_key, "") != "" for row in ordered):
            stds = [float(row[std_key]) for row in ordered]
            lower = [y - std for y, std in zip(ys, stds)]
            upper = [y + std for y, std in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)

    ax.set_xlabel("Partikelanzahl")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_frame_errors(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_id"],
            row["path_id"],
            row["localization_mode"],
            row.get("prior_case_index", ""),
            row["splat_id"],
            row.get("training_iteration", ""),
            row.get("particle_count", ""),
            int(float(row["frame_index"])),
        )
        grouped[key].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        scenario_id, path_id, mode, prior_case_index, splat_id, training_iteration, particle_count, frame_index = key
        combined_values = [to_float(item, "combined_pose_error_m") for item in group_rows]
        translation_values = [to_float(item, "translation_error_m") for item in group_rows]
        yaw_values = [to_float(item, "yaw_error_degrees") for item in group_rows]
        best_metric_values = [to_float(item, "best_metric_value", to_float(item, "best_score")) for item in group_rows]
        median_metric_values = [to_float(item, "median_metric_value") for item in group_rows]
        replay_times = [to_float(item, "replay_time_s") for item in group_rows]
        summaries.append(
            {
                "scenario_id": scenario_id,
                "path_id": path_id,
                "localization_mode": mode,
                "prior_case_index": prior_case_index,
                "splat_id": splat_id,
                "training_iteration": training_iteration,
                "particle_count": particle_count,
                "frame_index": frame_index,
                "mean_replay_time_s": mean(replay_times),
                "mean_combined_pose_error_m": mean(combined_values),
                "std_combined_pose_error_m": stdev(combined_values),
                "mean_translation_error_m": mean(translation_values),
                "mean_yaw_error_degrees": mean(yaw_values),
                "mean_best_metric_value": mean(best_metric_values),
                "std_best_metric_value": stdev(best_metric_values),
                "mean_median_metric_value": mean(median_metric_values),
                "sample_count": len(group_rows),
            }
        )

    summaries.sort(
        key=lambda row: (
            row["scenario_id"],
            row["path_id"],
            row["localization_mode"],
            int(float(row["prior_case_index"])) if row["prior_case_index"] else -1,
            int(float(row["training_iteration"])) if row["training_iteration"] else -1,
            int(float(row["particle_count"])) if row["particle_count"] else -1,
            int(row["frame_index"]),
        )
    )
    return summaries


def safe_name(*parts: str) -> str:
    return "__".join(part.replace("/", "_").replace(" ", "_") for part in parts if part != "")


def path_ids_by_scenario(rows: list[dict]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scenario_id = row.get("scenario_id", "")
        path_id = row.get("path_id", "")
        if scenario_id and path_id:
            grouped[scenario_id].add(path_id)
    return grouped


def plot_title(*, scenario_id: str, path_id: str, mode: str, prior_case_index: str, scenario_paths: dict[str, set[str]]) -> str:
    mode_label = "lokal" if mode == "local" else "global" if mode == "global" else mode
    prior_label = f" Prior {prior_case_index}" if mode == "local" and prior_case_index != "" else ""
    parts = [display_name(scenario_id)]
    if len(scenario_paths.get(scenario_id, set())) > 1:
        parts.append(display_name(path_id))
    parts.append(f"{mode_label}{prior_label}")
    return " / ".join(parts)


def output_name_parts(
    *, prefix: str, scenario_id: str, path_id: str, mode: str, prior_case_index: str, scenario_paths: dict[str, set[str]]
) -> list[str]:
    parts = [prefix, scenario_id]
    if len(scenario_paths.get(scenario_id, set())) > 1:
        parts.append(path_id)
    parts.append(mode)
    if prior_case_index != "":
        parts.append(f"prior{prior_case_index}")
    return parts


def particle_counts_by_run_scope(rows: list[dict]) -> dict[tuple[str, str, str, str], set[str]]:
    grouped: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[
            (
                row.get("scenario_id", ""),
                row.get("path_id", ""),
                row.get("localization_mode", ""),
                row.get("prior_case_index", ""),
            )
        ].add(row.get("particle_count", ""))
    return grouped


def particle_title_suffix(
    *, scenario_id: str, path_id: str, mode: str, prior_case_index: str, particle_count: str, scope_counts: dict[tuple[str, str, str, str], set[str]]
) -> str:
    key = (scenario_id, path_id, mode, prior_case_index)
    if len(scope_counts.get(key, set())) <= 1 or particle_count == "":
        return ""
    return f" / {int(float(particle_count))} Partikel"


def particle_output_suffix(
    *, scenario_id: str, path_id: str, mode: str, prior_case_index: str, particle_count: str, scope_counts: dict[tuple[str, str, str, str], set[str]]
) -> str:
    key = (scenario_id, path_id, mode, prior_case_index)
    if len(scope_counts.get(key, set())) <= 1 or particle_count == "":
        return ""
    return f"particles{int(float(particle_count))}"


def plot_convergence_by_frame(rows: list[dict], *, plots_dir: Path) -> None:
    scenario_paths = path_ids_by_scenario(rows)
    scope_counts = particle_counts_by_run_scope(rows)
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_id"],
            row["path_id"],
            row["localization_mode"],
            row.get("prior_case_index", ""),
            row.get("particle_count", ""),
        )
        grouped[key].append(row)

    for (scenario_id, path_id, mode, prior_case_index, particle_count), group_rows in sorted(grouped.items()):
        by_splat: dict[str, list[dict]] = defaultdict(list)
        for row in group_rows:
            by_splat[row["splat_id"]].append(row)

        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        for series_index, (splat_id, splat_rows) in enumerate(sorted(
            by_splat.items(),
            key=lambda item: int(float(item[1][0]["training_iteration"])) if item[1][0]["training_iteration"] else -1,
        )):
            ordered = sorted(splat_rows, key=lambda row: int(row["frame_index"]))
            xs = [int(row["frame_index"]) for row in ordered]
            ys = [float(row["mean_combined_pose_error_m"]) for row in ordered]
            iteration = ordered[0].get("training_iteration", "")
            label = f"{int(float(iteration))}" if iteration else display_name(splat_id)
            color = COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)]
            ax.plot(
                xs,
                ys,
                marker=MARKERS[series_index % len(MARKERS)],
                linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
                color=color,
                markersize=3,
                linewidth=1.5,
                label=label,
            )
            if all(row.get("std_combined_pose_error_m", "") != "" for row in ordered):
                stds = [float(row["std_combined_pose_error_m"]) for row in ordered]
                lower = [y - std for y, std in zip(ys, stds)]
                upper = [y + std for y, std in zip(ys, stds)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.12, linewidth=0)

        ax.set_title(
            plot_title(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                scenario_paths=scenario_paths,
            )
            + particle_title_suffix(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                particle_count=particle_count,
                scope_counts=scope_counts,
            )
        )
        ax.set_xlabel("Ausgewerteter PF-Frame")
        ax.set_ylabel("Mittlerer kombinierter Pose-Fehler [m]")
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="Splat-Iteration")
        output_base = plots_dir / safe_name(
            *output_name_parts(
                prefix="convergence_error_by_frame",
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                scenario_paths=scenario_paths,
            ),
            particle_output_suffix(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                particle_count=particle_count,
                scope_counts=scope_counts,
            ),
        )
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_best_metric_by_frame(rows: list[dict], *, plots_dir: Path) -> None:
    scenario_paths = path_ids_by_scenario(rows)
    scope_counts = particle_counts_by_run_scope(rows)
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_id"],
            row["path_id"],
            row["localization_mode"],
            row.get("prior_case_index", ""),
            row.get("particle_count", ""),
        )
        grouped[key].append(row)

    for (scenario_id, path_id, mode, prior_case_index, particle_count), group_rows in sorted(grouped.items()):
        by_splat: dict[str, list[dict]] = defaultdict(list)
        for row in group_rows:
            by_splat[row["splat_id"]].append(row)

        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        for series_index, (splat_id, splat_rows) in enumerate(sorted(
            by_splat.items(),
            key=lambda item: int(float(item[1][0]["training_iteration"])) if item[1][0]["training_iteration"] else -1,
        )):
            ordered = sorted(splat_rows, key=lambda row: int(row["frame_index"]))
            xs = [int(row["frame_index"]) for row in ordered]
            ys = [float(row["mean_best_metric_value"]) for row in ordered]
            iteration = ordered[0].get("training_iteration", "")
            label = f"{int(float(iteration))}" if iteration else display_name(splat_id)
            color = COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)]
            ax.plot(
                xs,
                ys,
                marker=MARKERS[series_index % len(MARKERS)],
                linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
                color=color,
                markersize=3,
                linewidth=1.5,
                label=label,
            )
            if all(row.get("std_best_metric_value", "") != "" for row in ordered):
                stds = [float(row["std_best_metric_value"]) for row in ordered]
                lower = [y - std for y, std in zip(ys, stds)]
                upper = [y + std for y, std in zip(ys, stds)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.12, linewidth=0)

        ax.set_title(
            plot_title(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                scenario_paths=scenario_paths,
            )
            + particle_title_suffix(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                particle_count=particle_count,
                scope_counts=scope_counts,
            )
        )
        ax.set_xlabel("Ausgewerteter PF-Frame")
        ax.set_ylabel("Mittlerer bester Metrikwert")
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="Splat-Iteration")
        output_base = plots_dir / safe_name(
            *output_name_parts(
                prefix="best_metric_by_frame",
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                scenario_paths=scenario_paths,
            ),
            particle_output_suffix(
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                particle_count=particle_count,
                scope_counts=scope_counts,
            ),
        )
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_frame_metric_by_particle_count(
    rows: list[dict],
    *,
    metric_key: str,
    ylabel: str,
    output_prefix: str,
    plots_dir: Path,
    std_key: str | None = None,
) -> None:
    scenario_paths = path_ids_by_scenario(rows)
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["scenario_id"],
                row["path_id"],
                row["localization_mode"],
                row.get("prior_case_index", ""),
                row["splat_id"],
            )
        ].append(row)

    for (scenario_id, path_id, mode, prior_case_index, splat_id), group_rows in sorted(grouped.items()):
        by_count: dict[str, list[dict]] = defaultdict(list)
        for row in group_rows:
            by_count[row.get("particle_count", "")].append(row)
        if len(by_count) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
        for series_index, (particle_count, count_rows) in enumerate(
            sorted(by_count.items(), key=lambda item: int(float(item[0])) if item[0] else -1)
        ):
            ordered = sorted(count_rows, key=lambda row: int(row["frame_index"]))
            xs = [int(row["frame_index"]) for row in ordered]
            ys = [float(row[metric_key]) for row in ordered]
            color = COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)]
            ax.plot(
                xs,
                ys,
                marker=MARKERS[series_index % len(MARKERS)],
                linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
                color=color,
                markersize=3,
                linewidth=1.5,
                label=f"{int(float(particle_count))} Partikel" if particle_count else "unbekannte Partikelanzahl",
            )
            if std_key is not None and all(row.get(std_key, "") != "" for row in ordered):
                stds = [float(row[std_key]) for row in ordered]
                lower = [y - std for y, std in zip(ys, stds)]
                upper = [y + std for y, std in zip(ys, stds)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.12, linewidth=0)

        title = plot_title(
            scenario_id=scenario_id,
            path_id=path_id,
            mode=mode,
            prior_case_index=prior_case_index,
            scenario_paths=scenario_paths,
        )
        iteration = group_rows[0].get("training_iteration", "")
        splat_label = f"{int(float(iteration))}" if iteration else display_name(splat_id)
        ax.set_title(f"{title} / {splat_label}")
        ax.set_xlabel("Ausgewerteter PF-Frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="Partikelanzahl")
        output_base = plots_dir / safe_name(
            *output_name_parts(
                prefix=output_prefix,
                scenario_id=scenario_id,
                path_id=path_id,
                mode=mode,
                prior_case_index=prior_case_index,
                scenario_paths=scenario_paths,
            ),
            splat_id,
        )
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    per_run_path = input_dir / "per_run_summary.csv"
    if not per_run_path.is_file():
        raise FileNotFoundError(f"Missing per-run summary: {per_run_path}")

    splat_iterations = set(args.splat_iterations) if args.splat_iterations else None
    summaries = summarize_by_condition(read_csv(per_run_path))
    plot_summaries = filter_by_splat_iterations(summaries, splat_iterations)
    write_csv(input_dir / "summary_by_condition.csv", summaries)

    plots_dir = input_dir / "plots"
    if splat_iterations:
        iteration_suffix = "_".join(str(iteration) for iteration in sorted(splat_iterations))
        plots_dir = plots_dir / f"iterations_{iteration_suffix}"
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_translation_error_m",
        ylabel="Mittlerer Translationsfehler [m]",
        output_path=plots_dir / "mean_translation_error_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_combined_pose_error_m",
        ylabel="Mittlerer kombinierter Pose-Fehler [m]",
        output_path=plots_dir / "combined_pose_error_by_splat",
    )
    # plot_metric_by_splat(
    #     summaries,
    #     metric_key="mean_failure_rate",
    #     std_key="std_failure_rate",
    #     ylabel="Fehlerrate",
    #     output_path=plots_dir / "failure_rate_by_splat",
    # )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_render_and_score_ms",
        ylabel="Mittlere Render- und Bewertungszeit [ms]",
        output_path=plots_dir / "runtime_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_total_hz",
        ylabel="Mittlerer PF-Durchsatz [Hz]",
        output_path=plots_dir / "total_hz_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_max_gpu_memory_used_mb",
        ylabel="Maximal genutzter GPU-Speicher [MiB]",
        output_path=plots_dir / "gpu_memory_by_splat",
    )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_mean_combined_pose_error_m",
        std_key="std_mean_combined_pose_error_m",
        ylabel="Mittlerer kombinierter Pose-Fehler [m]",
        output_path=plots_dir / "combined_pose_error_by_particle_count",
    )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_p95_combined_pose_error_m",
        ylabel="P95 kombinierter Pose-Fehler [m]",
        output_path=plots_dir / "p95_combined_pose_error_by_particle_count",
    )
    # plot_metric_by_particle_count(
    #     plot_summaries,
    #     metric_key="mean_failure_rate",
    #     std_key="std_failure_rate",
    #     ylabel="Fehlerrate",
    #     output_path=plots_dir / "failure_rate_by_particle_count",
    # )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_converged",
        ylabel="Konvergenzrate",
        output_path=plots_dir / "convergence_rate_by_particle_count",
    )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_mean_total_hz",
        ylabel="Mittlerer PF-Durchsatz [Hz]",
        output_path=plots_dir / "total_hz_by_particle_count",
    )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_mean_render_and_score_ms",
        ylabel="Mittlere Render- und Bewertungszeit [ms]",
        output_path=plots_dir / "runtime_by_particle_count",
    )
    plot_metric_by_particle_count(
        plot_summaries,
        metric_key="mean_max_gpu_memory_used_mb",
        ylabel="Maximal genutzter GPU-Speicher [MiB]",
        output_path=plots_dir / "gpu_memory_by_particle_count",
    )
    per_frame_path = input_dir / "per_frame.csv"
    if per_frame_path.is_file():
        frame_error_summaries = summarize_frame_errors(read_csv(per_frame_path))
        plot_frame_error_summaries = filter_by_splat_iterations(frame_error_summaries, splat_iterations)
        write_csv(input_dir / "frame_error_by_condition.csv", frame_error_summaries)
        plot_convergence_by_frame(plot_frame_error_summaries, plots_dir=plots_dir)
        plot_best_metric_by_frame(plot_frame_error_summaries, plots_dir=plots_dir)
        plot_frame_metric_by_particle_count(
            plot_frame_error_summaries,
            metric_key="mean_combined_pose_error_m",
            std_key="std_combined_pose_error_m",
            ylabel="Mittlerer kombinierter Pose-Fehler [m]",
            output_prefix="convergence_error_by_frame_by_particle_count",
            plots_dir=plots_dir,
        )
        plot_frame_metric_by_particle_count(
            plot_frame_error_summaries,
            metric_key="mean_best_metric_value",
            std_key="std_best_metric_value",
            ylabel="Mittlerer bester Metrikwert",
            output_prefix="best_metric_by_frame_by_particle_count",
            plots_dir=plots_dir,
        )
    print(f"Wrote {input_dir / 'summary_by_condition.csv'}")
    if per_frame_path.is_file():
        print(f"Wrote {input_dir / 'frame_error_by_condition.csv'}")
    print(f"Wrote plots under {plots_dir}")


if __name__ == "__main__":
    main()
