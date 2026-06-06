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
            row["splat_id"],
            row.get("training_iteration", ""),
            row.get("particle_count", ""),
            row.get("metric_name", ""),
        )
        grouped[key].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        scenario_id, mode, splat_id, training_iteration, particle_count, metric_name = key
        row = {
            "scenario_id": scenario_id,
            "localization_mode": mode,
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
        grouped[(row["scenario_id"], row["localization_mode"])].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for (scenario_id, mode), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(float(row["training_iteration"])))
        xs = [int(float(row["training_iteration"])) for row in ordered]
        ys = [float(row[metric_key]) for row in ordered]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"{scenario_id} / {mode}")

    ax.set_xlabel("Splat training iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
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
    print(f"Wrote {input_dir / 'summary_by_condition.csv'}")
    print(f"Wrote plots under {plots_dir}")


if __name__ == "__main__":
    main()
