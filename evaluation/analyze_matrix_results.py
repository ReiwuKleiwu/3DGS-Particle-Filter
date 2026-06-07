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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plot_style import COLORBLIND_SAFE_COLORS, LINE_STYLES, MARKERS


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
    parser = argparse.ArgumentParser(description="Analyze outputs from evaluation/run_matrix_experiment.py.")
    parser.add_argument("--input-dir", type=Path, required=True)
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


def plot_metric_by_splat(rows: list[dict], *, metric_key: str, ylabel: str, output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], row["localization_mode"], row.get("prior_case_index", ""))].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for series_index, ((scenario_id, mode, prior_case_index), group_rows) in enumerate(sorted(grouped.items())):
        ordered = sorted(group_rows, key=lambda row: int(float(row["training_iteration"])))
        xs = [int(float(row["training_iteration"])) for row in ordered]
        ys = [float(row[metric_key]) for row in ordered]
        prior_label = f" / prior {prior_case_index}" if mode == "local" and prior_case_index != "" else ""
        ax.plot(
            xs,
            ys,
            marker=MARKERS[series_index % len(MARKERS)],
            linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
            color=COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)],
            linewidth=1.7,
            label=f"{scenario_id} / {mode}{prior_label}",
        )

    ax.set_xlabel("Splat training iteration")
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
            int(float(row["frame_index"])),
        )
        grouped[key].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        scenario_id, path_id, mode, prior_case_index, splat_id, training_iteration, frame_index = key
        combined_values = [to_float(item, "combined_pose_error_m") for item in group_rows]
        translation_values = [to_float(item, "translation_error_m") for item in group_rows]
        yaw_values = [to_float(item, "yaw_error_degrees") for item in group_rows]
        replay_times = [to_float(item, "replay_time_s") for item in group_rows]
        summaries.append(
            {
                "scenario_id": scenario_id,
                "path_id": path_id,
                "localization_mode": mode,
                "prior_case_index": prior_case_index,
                "splat_id": splat_id,
                "training_iteration": training_iteration,
                "frame_index": frame_index,
                "mean_replay_time_s": mean(replay_times),
                "mean_combined_pose_error_m": mean(combined_values),
                "std_combined_pose_error_m": stdev(combined_values),
                "mean_translation_error_m": mean(translation_values),
                "mean_yaw_error_degrees": mean(yaw_values),
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
            int(row["frame_index"]),
        )
    )
    return summaries


def safe_name(*parts: str) -> str:
    return "__".join(part.replace("/", "_").replace(" ", "_") for part in parts if part != "")


def plot_convergence_by_frame(rows: list[dict], *, plots_dir: Path) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario_id"],
            row["path_id"],
            row["localization_mode"],
            row.get("prior_case_index", ""),
        )
        grouped[key].append(row)

    for (scenario_id, path_id, mode, prior_case_index), group_rows in sorted(grouped.items()):
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
            label = f"{int(float(iteration))} iter" if iteration else splat_id
            ax.plot(
                xs,
                ys,
                marker=MARKERS[series_index % len(MARKERS)],
                linestyle=LINE_STYLES[series_index % len(LINE_STYLES)],
                color=COLORBLIND_SAFE_COLORS[series_index % len(COLORBLIND_SAFE_COLORS)],
                markersize=3,
                linewidth=1.5,
                label=label,
            )

        prior_label = f" prior {prior_case_index}" if mode == "local" and prior_case_index != "" else ""
        ax.set_title(f"{scenario_id} / {path_id} / {mode}{prior_label}")
        ax.set_xlabel("Evaluated PF frame")
        ax.set_ylabel("Mean combined pose error [m]")
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="Splat")
        output_base = plots_dir / safe_name("convergence_error_by_frame", scenario_id, path_id, mode, f"prior{prior_case_index}" if prior_case_index != "" else "")
        fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    per_run_path = input_dir / "per_run_summary.csv"
    if not per_run_path.is_file():
        raise FileNotFoundError(f"Missing per-run summary: {per_run_path}")

    summaries = summarize_by_condition(read_csv(per_run_path))
    write_csv(input_dir / "summary_by_condition.csv", summaries)

    plots_dir = input_dir / "plots"
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_combined_pose_error_m",
        ylabel="Mean combined pose error [m]",
        output_path=plots_dir / "combined_pose_error_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_failure_rate",
        ylabel="Failure rate",
        output_path=plots_dir / "failure_rate_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_render_and_score_ms",
        ylabel="Mean render and score time [ms]",
        output_path=plots_dir / "runtime_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_mean_total_hz",
        ylabel="Mean total PF throughput [Hz]",
        output_path=plots_dir / "total_hz_by_splat",
    )
    plot_metric_by_splat(
        summaries,
        metric_key="mean_max_gpu_memory_used_mb",
        ylabel="Max GPU memory used [MiB]",
        output_path=plots_dir / "gpu_memory_by_splat",
    )
    per_frame_path = input_dir / "per_frame.csv"
    if per_frame_path.is_file():
        frame_error_summaries = summarize_frame_errors(read_csv(per_frame_path))
        write_csv(input_dir / "frame_error_by_condition.csv", frame_error_summaries)
        plot_convergence_by_frame(frame_error_summaries, plots_dir=plots_dir)
    print(f"Wrote {input_dir / 'summary_by_condition.csv'}")
    if per_frame_path.is_file():
        print(f"Wrote {input_dir / 'frame_error_by_condition.csv'}")
    print(f"Wrote plots under {plots_dir}")


if __name__ == "__main__":
    main()
