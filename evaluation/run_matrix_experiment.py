#!/usr/bin/env python3
"""Run matrix-based replay localization experiments for thesis evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for minimal environments
    Console = None
    Progress = None
    RICH_AVAILABLE = False

from core.config import MeasurementSettings, load_turtlebot_localization_config
from core.particle_filter.application.step_engine import LocalizationStepEngine
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from core.particle_filter.domain.pose import Pose2D, wrap_angle
from core.particle_filter.domain.recovery import AugmentedMclRecoveryTracker
from core.particle_filter.infrastructure.map import FreeSpacePoseSampler
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from evaluation.evaluator import build_observation
from evaluation.models import PriorOffset, ReplayManifest
from evaluation.paths import RESULTS_DIR


DEFAULT_SPLAT_ITERATIONS = [1000, 2000, 3000, 5000, 8000, 12000, 18000, 30000]
DEFAULT_SEEDS = [1001, 1002, 1003, 1004, 1005]


@dataclass(frozen=True)
class SplatSpec:
    splat_id: str
    ply_path: Path
    training_iteration: int | None
    quality_label: str
    notes: str


@dataclass(frozen=True)
class PathSpec:
    scenario_id: str
    path_id: str
    manifest_path: Path
    splat_csv: Path


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    particle_count: int
    prior_case_index: int
    prior_offset: PriorOffset | None


@dataclass(frozen=True)
class Thresholds:
    failure_translation_m: float = 0.5
    failure_yaw_deg: float = 20.0
    lost_translation_m: float = 0.75
    lost_yaw_deg: float = 30.0
    convergence_translation_m: float = 0.25
    convergence_yaw_deg: float = 10.0
    convergence_consecutive_frames: int = 5


@dataclass(frozen=True)
class ParticleRecordingConfig:
    enabled: bool = False
    frame_stride: int = 1
    modes: frozenset[str] | None = None
    path_ids: frozenset[str] | None = None
    splat_ids: frozenset[str] | None = None
    seeds: frozenset[int] | None = None

    def matches(self, *, mode: str, path_id: str, splat_id: str, seed: int) -> bool:
        if not self.enabled:
            return False
        return (
            (self.modes is None or mode in self.modes)
            and (self.path_ids is None or path_id in self.path_ids)
            and (self.splat_ids is None or splat_id in self.splat_ids)
            and (self.seeds is None or seed in self.seeds)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario/path/splat/mode/seed replay experiments.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("turtlebot_localization.yaml"))
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable rich progress bars and use plain logs.")
    return parser.parse_args()


def resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        matrix_relative = (base_dir / path).resolve()
        repo_relative = (REPO_ROOT / path).resolve()
        path = matrix_relative if matrix_relative.exists() else repo_relative
    return path.resolve()


def load_matrix(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    if not isinstance(matrix, dict):
        raise ValueError("Experiment matrix must be a YAML mapping")
    return matrix


def load_splats(path: Path) -> list[SplatSpec]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No splats listed in {path}")

    splats = []
    seen_ids = set()
    for row in rows:
        splat_id = str(row.get("splat_id", "")).strip()
        if not splat_id:
            raise ValueError(f"Missing splat_id in {path}")
        if splat_id in seen_ids:
            raise ValueError(f"Duplicate splat_id in {path}: {splat_id}")
        seen_ids.add(splat_id)

        ply_path = Path(str(row.get("ply_path", "")).strip()).expanduser().resolve()
        if not ply_path.is_file():
            raise FileNotFoundError(f"Splat file not found for {splat_id}: {ply_path}")
        iteration = str(row.get("training_iteration", "")).strip()
        splats.append(
            SplatSpec(
                splat_id=splat_id,
                ply_path=ply_path,
                training_iteration=int(iteration) if iteration else None,
                quality_label=str(row.get("quality_label", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
        )
    return splats


def selected_splats(splats: list[SplatSpec], iterations: list[int] | None) -> list[SplatSpec]:
    if not iterations:
        return splats
    wanted = set(iterations)
    selected = [splat for splat in splats if splat.training_iteration in wanted]
    missing = sorted(wanted - {splat.training_iteration for splat in selected if splat.training_iteration is not None})
    if missing:
        raise ValueError(f"Splat CSV is missing requested iterations: {missing}")
    return selected


def parse_modes(raw_modes: dict) -> list[ModeSpec]:
    modes = []
    for mode_name, raw_mode in raw_modes.items():
        mode = str(mode_name).strip().lower()
        if mode not in {"local", "global"}:
            raise ValueError(f"Unsupported mode in matrix: {mode_name}")
        particle_count = int(raw_mode.get("particle_count", 500 if mode == "local" else 2000))
        if mode == "local":
            raw_offsets = raw_mode.get("prior_offsets")
            if raw_offsets is None:
                raw_offsets = [raw_mode.get("prior_offset", {"dx": 0.40, "dy": 0.0, "dyaw_degrees": 10.0})]
            if not isinstance(raw_offsets, list) or not raw_offsets:
                raise ValueError("modes.local.prior_offsets must be a non-empty list")
            for prior_case_index, raw_offset in enumerate(raw_offsets):
                modes.append(
                    ModeSpec(
                        mode=mode,
                        particle_count=particle_count,
                        prior_case_index=prior_case_index,
                        prior_offset=PriorOffset(
                            dx=float(raw_offset.get("dx", 0.40)),
                            dy=float(raw_offset.get("dy", 0.0)),
                            dyaw_degrees=float(raw_offset.get("dyaw_degrees", 10.0)),
                        ),
                    )
                )
        else:
            modes.append(ModeSpec(mode=mode, particle_count=particle_count, prior_case_index=0, prior_offset=None))
    if not modes:
        raise ValueError("Matrix must define at least one mode")
    return modes


def parse_path_specs(matrix: dict, *, base_dir: Path) -> list[PathSpec]:
    path_specs = []
    for scenario in matrix.get("scenarios", []):
        scenario_id = str(scenario["scenario_id"]).strip()
        splat_csv = resolve_path(scenario["splat_csv"], base_dir=base_dir)
        if not splat_csv.is_file():
            raise FileNotFoundError(f"Missing splat CSV for scenario {scenario_id}: {splat_csv}")
        for path in scenario.get("paths", []):
            manifest_path = resolve_path(path["manifest"], base_dir=base_dir)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing manifest for scenario {scenario_id}: {manifest_path}")
            path_specs.append(
                PathSpec(
                    scenario_id=scenario_id,
                    path_id=str(path["path_id"]).strip(),
                    manifest_path=manifest_path,
                    splat_csv=splat_csv,
                )
            )
    if not path_specs:
        raise ValueError("Matrix must define at least one scenario path")
    return path_specs


def optional_str_set(raw: object, *, key: str) -> frozenset[str] | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, list):
        raise ValueError(f"record_particles.{key} must be a list")
    return frozenset(str(value).strip() for value in raw if str(value).strip())


def optional_int_set(raw: object, *, key: str) -> frozenset[int] | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, list):
        raise ValueError(f"record_particles.{key} must be a list")
    return frozenset(int(value) for value in raw)


def parse_particle_recording(raw: object) -> ParticleRecordingConfig:
    if raw is None:
        return ParticleRecordingConfig()
    if isinstance(raw, bool):
        return ParticleRecordingConfig(enabled=raw)
    if not isinstance(raw, dict):
        raise ValueError("experiment.record_particles must be a boolean or mapping")

    frame_stride = int(raw.get("frame_stride", 1))
    if frame_stride <= 0:
        raise ValueError("record_particles.frame_stride must be positive")
    modes = optional_str_set(raw.get("modes"), key="modes")
    if modes is not None:
        invalid_modes = modes - {"local", "global"}
        if invalid_modes:
            raise ValueError(f"record_particles.modes contains unsupported modes: {sorted(invalid_modes)}")
    return ParticleRecordingConfig(
        enabled=bool(raw.get("enabled", True)),
        frame_stride=frame_stride,
        modes=modes,
        path_ids=optional_str_set(raw.get("path_ids"), key="path_ids"),
        splat_ids=optional_str_set(raw.get("splat_ids"), key="splat_ids"),
        seeds=optional_int_set(raw.get("seeds"), key="seeds"),
    )


def parse_thresholds(raw: dict | None) -> Thresholds:
    raw = raw or {}
    return Thresholds(
        failure_translation_m=float(raw.get("failure_translation_m", 0.5)),
        failure_yaw_deg=float(raw.get("failure_yaw_deg", 20.0)),
        lost_translation_m=float(raw.get("lost_translation_m", 0.75)),
        lost_yaw_deg=float(raw.get("lost_yaw_deg", 30.0)),
        convergence_translation_m=float(raw.get("convergence_translation_m", 0.25)),
        convergence_yaw_deg=float(raw.get("convergence_yaw_deg", 10.0)),
        convergence_consecutive_frames=int(raw.get("convergence_consecutive_frames", 5)),
    )


def measurement_with_overrides(base: MeasurementSettings, raw: dict | None) -> MeasurementSettings:
    if not raw:
        return base
    allowed = {
        "metric_name",
        "temperature",
        "packed",
        "radius_clip",
        "hybrid_ssim_weight",
        "hybrid_l1_weight",
        "hybrid_gradient_weight",
        "lpips_top_k",
        "lpips_weight",
        "lpips_net",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported measurement override keys: {sorted(unknown)}")
    return replace(base, **raw)


def restart_renderer(*, splat: SplatSpec, backend: str, port: int, build_image: bool, settings=None) -> None:
    env = os.environ.copy()
    env.update(
        {
            "SPLAT_PATH": str(splat.ply_path),
            "BACKEND": backend,
            "PORT": str(port),
            "BUILD_IMAGE": "1" if build_image else "0",
        }
    )
    if settings is not None:
        env.update(
            {
                "SPLAT_MAP_X": str(settings.renderer.splat_map_x),
                "SPLAT_MAP_Y": str(settings.renderer.splat_map_y),
                "SPLAT_MAP_SCALE": str(settings.renderer.splat_map_scale),
                "SPLAT_MAP_SCALE_X": str(
                    settings.renderer.splat_map_scale_x
                    if settings.renderer.splat_map_scale_x is not None
                    else settings.renderer.splat_map_scale
                ),
                "SPLAT_MAP_SCALE_Y": str(
                    settings.renderer.splat_map_scale_y
                    if settings.renderer.splat_map_scale_y is not None
                    else settings.renderer.splat_map_scale
                ),
                "SPLAT_MAP_YAW_DEGREES": str(settings.renderer.splat_map_yaw_degrees),
            }
        )
    print(f"Restarting renderer for {splat.splat_id}: {splat.ply_path}", flush=True)
    subprocess.run([str(REPO_ROOT / "start_renderer.sh")], cwd=REPO_ROOT, env=env, check=True)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def query_gpu_memory() -> dict[str, float | str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {
            "gpu_index": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
            "gpu_memory_free_mb": "",
        }

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {
            "gpu_index": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
            "gpu_memory_free_mb": "",
        }
    first_row = [part.strip() for part in rows[0].split(",")]
    if len(first_row) < 3:
        return {
            "gpu_index": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
            "gpu_memory_free_mb": "",
        }
    used_mb = float(first_row[1])
    total_mb = float(first_row[2])
    return {
        "gpu_index": first_row[0],
        "gpu_memory_used_mb": used_mb,
        "gpu_memory_total_mb": total_mb,
        "gpu_memory_free_mb": total_mb - used_mb,
    }


PARTICLE_SNAPSHOT_COLUMNS = [
    "run_id",
    "scenario_id",
    "path_id",
    "splat_id",
    "training_iteration",
    "seed",
    "localization_mode",
    "prior_case_index",
    "frame_index",
    "replay_time_s",
    "particle_index",
    "x",
    "y",
    "yaw",
    "weight",
    "recovery_sample",
    "roughening_sample",
]


def write_particle_snapshot(
    writer: csv.DictWriter,
    *,
    run_id: str,
    scenario_id: str,
    path_id: str,
    splat: SplatSpec,
    mode: ModeSpec,
    seed: int,
    frame_index: int,
    replay_time_s: float,
    particles,
) -> None:
    for particle_index, particle in enumerate(particles):
        writer.writerow(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "path_id": path_id,
                "splat_id": splat.splat_id,
                "training_iteration": splat.training_iteration,
                "seed": seed,
                "localization_mode": mode.mode,
                "prior_case_index": mode.prior_case_index,
                "frame_index": frame_index,
                "replay_time_s": replay_time_s,
                "particle_index": particle_index,
                "x": particle.pose.x,
                "y": particle.pose.y,
                "yaw": particle.pose.yaw,
                "weight": particle.weight,
                "recovery_sample": int(particle.recovery_sample),
                "roughening_sample": int(particle.roughening_sample),
            }
        )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def consecutive_true_start(values: list[bool], required_count: int) -> int | None:
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if value else 0
        if streak >= required_count:
            return index
    return None


def replay_time_seconds(frame, first_stamp_ns: int | None, frame_index: int) -> float:
    stamp_ns = frame.image_stamp_seconds * 1_000_000_000 + frame.image_stamp_nanoseconds
    if first_stamp_ns is None or stamp_ns == 0:
        return float(frame_index)
    return max(0.0, (stamp_ns - first_stamp_ns) / 1_000_000_000)


def pose_error(estimated: Pose2D, truth: Pose2D) -> dict:
    x_error = float(estimated.x - truth.x)
    y_error = float(estimated.y - truth.y)
    yaw_error_rad = float(abs(wrap_angle(estimated.yaw - truth.yaw)))
    translation_error = float(math.hypot(x_error, y_error))
    return {
        "x_error_m": x_error,
        "y_error_m": y_error,
        "translation_error_m": translation_error,
        "yaw_error_rad": yaw_error_rad,
        "yaw_error_degrees": float(math.degrees(yaw_error_rad)),
        "combined_pose_error_m": float(translation_error + 0.5 * yaw_error_rad),
    }


def max_consecutive(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def build_particle_filter(*, settings, mode: ModeSpec, manifest: ReplayManifest, rng: random.Random, global_sampler):
    motion_model = TurtleBotMotionModel(
        noise_x=settings.motion_noise.x_meters,
        noise_y=settings.motion_noise.y_meters,
        noise_yaw=settings.motion_noise.yaw_radians,
        rng=rng,
    )
    particle_filter = TurtleBotParticleFilter(
        config=TurtleBotParticleFilterConfig(
            particle_count=mode.particle_count,
            resample_threshold_ratio=settings.particle_filter.resample_threshold_ratio,
            roughening_enabled=settings.particle_filter.roughening_enabled,
            roughening_mode=settings.particle_filter.roughening_mode,
            roughening_ratio=settings.particle_filter.roughening_ratio,
            roughening_sigma_x=settings.particle_filter.roughening_sigma_x,
            roughening_sigma_y=settings.particle_filter.roughening_sigma_y,
            roughening_sigma_yaw=settings.particle_filter.roughening_sigma_yaw,
        ),
        motion_model=motion_model,
        rng=rng,
    )
    if mode.mode == "global":
        particle_filter.initialize_global(global_sampler)
    else:
        if mode.prior_offset is None:
            raise ValueError("Local mode requires a prior offset")
        prior = mode.prior_offset.apply(
            manifest.frames[0].pose,
            sigma_x=manifest.initial_prior.sigma_x,
            sigma_y=manifest.initial_prior.sigma_y,
            sigma_yaw_degrees=math.degrees(manifest.initial_prior.sigma_yaw),
        )
        particle_filter.initialize(prior)
    return particle_filter


def make_progress(enabled: bool):
    if not enabled or not RICH_AVAILABLE:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )


def print_study_summary(
    *,
    console,
    run_name: str,
    path_specs: list[PathSpec],
    splats_by_csv: dict[Path, list[SplatSpec]],
    modes: list[ModeSpec],
    seeds: list[int],
    frame_stride: int,
    total_runs: int,
    estimated_total_frames: int,
) -> None:
    local_modes = [mode for mode in modes if mode.mode == "local"]
    global_modes = [mode for mode in modes if mode.mode == "global"]
    scenario_count = len({path.scenario_id for path in path_specs})
    path_count = len(path_specs)
    splat_count = sum(len(splats_by_csv[path.splat_csv]) for path in path_specs)
    mean_splats_per_path = splat_count / max(path_count, 1)

    if RICH_AVAILABLE and console is not None:
        table = Table(title="Evaluation study")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("run_name", run_name)
        table.add_row("scenarios", str(scenario_count))
        table.add_row("paths", str(path_count))
        table.add_row("splats/path", f"{mean_splats_per_path:.0f}")
        table.add_row("seeds", ", ".join(str(seed) for seed in seeds))
        table.add_row("local prior cases", str(len(local_modes)))
        table.add_row("global modes", str(len(global_modes)))
        table.add_row("frame_stride", str(frame_stride))
        table.add_row("total runs", str(total_runs))
        table.add_row("estimated evaluated frames", str(estimated_total_frames))
        console.print(table)
        return

    print("Evaluation study")
    print(f"  run_name: {run_name}")
    print(f"  scenarios: {scenario_count}")
    print(f"  paths: {path_count}")
    print(f"  splats/path: {mean_splats_per_path:.0f}")
    print(f"  seeds: {seeds}")
    print(f"  local prior cases: {len(local_modes)}")
    print(f"  global modes: {len(global_modes)}")
    print(f"  frame_stride: {frame_stride}")
    print(f"  total runs: {total_runs}")
    print(f"  estimated evaluated frames: {estimated_total_frames}")


def run_label(
    *,
    scenario_id: str,
    path_id: str,
    splat_id: str,
    mode: ModeSpec,
    seed: int,
) -> str:
    prior = f" prior={mode.prior_case_index}" if mode.mode == "local" else ""
    return f"{scenario_id}/{path_id} {splat_id} {mode.mode}{prior} seed={seed}"


def evaluate_run(
    *,
    run_id: str,
    scenario_id: str,
    path_id: str,
    splat: SplatSpec,
    mode: ModeSpec,
    seed: int,
    manifest: ReplayManifest,
    manifest_path: Path,
    renderer_client: RendererServiceClient,
    settings,
    measurement: MeasurementSettings,
    frame_stride: int,
    thresholds: Thresholds,
    global_sampler,
    output_dir: Path,
    particle_recording: ParticleRecordingConfig,
    gpu_memory_poll_stride: int,
    progress=None,
    run_task_id=None,
    run_label_text: str = "",
) -> tuple[list[dict], dict]:
    sampled_frames = manifest.frames[::frame_stride]
    if not sampled_frames:
        raise ValueError(f"No frames selected for {manifest_path} with frame_stride={frame_stride}")
    if progress is not None and run_task_id is not None:
        progress.update(
            run_task_id,
            total=len(sampled_frames),
            completed=0,
            description=f"frames {run_label_text}",
            visible=True,
        )

    rng = random.Random(seed)
    particle_filter = build_particle_filter(
        settings=settings,
        mode=mode,
        manifest=manifest,
        rng=rng,
        global_sampler=global_sampler,
    )
    step_engine = LocalizationStepEngine(renderer_client, measurement)
    recovery_tracker = AugmentedMclRecoveryTracker(settings.recovery) if settings.recovery.enabled else None

    first_stamp_ns = sampled_frames[0].image_stamp_seconds * 1_000_000_000 + sampled_frames[0].image_stamp_nanoseconds
    if first_stamp_ns == 0:
        first_stamp_ns = None

    previous_odom_pose = None
    frame_rows = []
    error_rows = []
    total_frame_ms_values = []
    render_ms_values = []
    gpu_memory_used_mb_values = []
    gpu_memory_free_mb_values = []
    ess_values = []
    resample_count = 0
    recovery_event_count = 0
    last_gpu_memory = {
        "gpu_index": "",
        "gpu_memory_used_mb": "",
        "gpu_memory_total_mb": "",
        "gpu_memory_free_mb": "",
    }
    should_record_particles = particle_recording.matches(
        mode=mode.mode,
        path_id=path_id,
        splat_id=splat.splat_id,
        seed=seed,
    )
    particle_file = None
    particle_writer = None
    if should_record_particles:
        particle_snapshot_path = output_dir / "particles" / f"{run_id}.csv"
        particle_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        particle_file = particle_snapshot_path.open("w", newline="", encoding="utf-8")
        particle_writer = csv.DictWriter(particle_file, fieldnames=PARTICLE_SNAPSHOT_COLUMNS)
        particle_writer.writeheader()

    try:
        for frame_index, frame in enumerate(sampled_frames):
            observation = build_observation(manifest, frame, frame_index, settings.camera_override)
            frame_start = time.perf_counter()
            step_result = step_engine.run_step(
                particle_filter=particle_filter,
                observation=observation,
                previous_odometry_pose=previous_odom_pose,
                recovery_tracker=recovery_tracker,
                random_pose_sampler=global_sampler,
            )
            total_frame_ms = (time.perf_counter() - frame_start) * 1000.0
            previous_odom_pose = step_result.previous_odometry_pose

            if frame_index % gpu_memory_poll_stride == 0:
                last_gpu_memory = query_gpu_memory()
            gpu_memory_used_mb = last_gpu_memory["gpu_memory_used_mb"]
            gpu_memory_free_mb = last_gpu_memory["gpu_memory_free_mb"]
            if gpu_memory_used_mb != "":
                gpu_memory_used_mb_values.append(float(gpu_memory_used_mb))
            if gpu_memory_free_mb != "":
                gpu_memory_free_mb_values.append(float(gpu_memory_free_mb))

            errors = pose_error(step_result.estimated_pose, frame.pose)
            is_failure = (
                errors["translation_error_m"] > thresholds.failure_translation_m
                or errors["yaw_error_degrees"] > thresholds.failure_yaw_deg
            )
            is_lost = (
                errors["translation_error_m"] > thresholds.lost_translation_m
                or errors["yaw_error_degrees"] > thresholds.lost_yaw_deg
            )
            is_convergence_candidate = (
                errors["translation_error_m"] < thresholds.convergence_translation_m
                and errors["yaw_error_degrees"] < thresholds.convergence_yaw_deg
            )

            render_ms = step_result.score_result.elapsed_milliseconds
            replay_time_s = replay_time_seconds(frame, first_stamp_ns, frame_index)
            total_frame_ms_values.append(total_frame_ms)
            render_ms_values.append(render_ms)
            ess_values.append(step_result.effective_particle_count)
            resample_count += int(step_result.resampled)
            recovery_event_count += int(step_result.random_particle_count > 0)
            error_rows.append(
                {
                    **errors,
                    "is_failure": is_failure,
                    "is_lost_tracking": is_lost,
                    "is_convergence_candidate": is_convergence_candidate,
                    "replay_time_s": replay_time_s,
                }
            )

            if (
                particle_writer is not None
                and frame_index % particle_recording.frame_stride == 0
            ):
                write_particle_snapshot(
                    particle_writer,
                    run_id=run_id,
                    scenario_id=scenario_id,
                    path_id=path_id,
                    splat=splat,
                    mode=mode,
                    seed=seed,
                    frame_index=frame_index,
                    replay_time_s=replay_time_s,
                    particles=particle_filter.particles,
                )

            frame_rows.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "path_id": path_id,
                    "splat_id": splat.splat_id,
                    "splat_path": str(splat.ply_path),
                    "training_iteration": splat.training_iteration,
                    "quality_label": splat.quality_label,
                    "manifest": str(manifest_path),
                    "seed": seed,
                    "localization_mode": mode.mode,
                    "particle_count": mode.particle_count,
                    "metric_name": measurement.metric_name,
                    "prior_case_index": mode.prior_case_index,
                    "prior_dx": mode.prior_offset.dx if mode.prior_offset else "",
                    "prior_dy": mode.prior_offset.dy if mode.prior_offset else "",
                    "prior_dyaw_degrees": mode.prior_offset.dyaw_degrees if mode.prior_offset else "",
                    "frame_index": frame_index,
                    "image_path": frame.image_path,
                    "frame_timestamp_s": frame.image_stamp_seconds + frame.image_stamp_nanoseconds / 1_000_000_000,
                    "replay_time_s": replay_time_s,
                    "truth_x": frame.pose.x,
                    "truth_y": frame.pose.y,
                    "truth_yaw": frame.pose.yaw,
                    "estimate_x": step_result.estimated_pose.x,
                    "estimate_y": step_result.estimated_pose.y,
                    "estimate_yaw": step_result.estimated_pose.yaw,
                    **errors,
                    "is_failure": int(is_failure),
                    "is_lost_tracking": int(is_lost),
                    "is_converged_frame": 0,
                    "effective_particle_count": step_result.effective_particle_count,
                    "resampled": int(step_result.resampled),
                    "best_particle_index": step_result.score_result.best_index,
                    "best_score": step_result.best_score,
                    "measurement_likelihood": step_result.measurement_likelihood,
                    "render_and_score_ms": render_ms,
                    "pf_step_ms": max(0.0, total_frame_ms - render_ms),
                    "total_frame_ms": total_frame_ms,
                    "total_hz": 1000.0 / total_frame_ms if total_frame_ms > 0.0 else "",
                    "gpu_index": last_gpu_memory["gpu_index"],
                    "gpu_memory_used_mb": gpu_memory_used_mb,
                    "gpu_memory_total_mb": last_gpu_memory["gpu_memory_total_mb"],
                    "gpu_memory_free_mb": gpu_memory_free_mb,
                    "random_particle_ratio": step_result.random_particle_ratio,
                    "random_particle_count": step_result.random_particle_count,
                    "roughening_particle_count": step_result.roughening_particle_count,
                }
            )
            if progress is not None and run_task_id is not None:
                progress.update(run_task_id, advance=1)
    finally:
        if particle_file is not None:
            particle_file.close()

    convergence_index = consecutive_true_start(
        [row["is_convergence_candidate"] for row in error_rows],
        thresholds.convergence_consecutive_frames,
    )
    if convergence_index is not None:
        for row in frame_rows[convergence_index:]:
            row["is_converged_frame"] = 1

    post_convergence_rows = error_rows[convergence_index:] if convergence_index is not None else []
    final = error_rows[-1]
    summary = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "path_id": path_id,
        "splat_id": splat.splat_id,
        "splat_path": str(splat.ply_path),
        "splat_file_size_mb": splat.ply_path.stat().st_size / (1024 * 1024),
        "training_iteration": splat.training_iteration,
        "quality_label": splat.quality_label,
        "manifest": str(manifest_path),
        "seed": seed,
        "localization_mode": mode.mode,
        "particle_count": mode.particle_count,
        "metric_name": measurement.metric_name,
        "prior_case_index": mode.prior_case_index,
        "prior_dx": mode.prior_offset.dx if mode.prior_offset else "",
        "prior_dy": mode.prior_offset.dy if mode.prior_offset else "",
        "prior_dyaw_degrees": mode.prior_offset.dyaw_degrees if mode.prior_offset else "",
        "frame_count": len(sampled_frames),
        "mean_translation_error_m": float(np.mean([row["translation_error_m"] for row in error_rows])),
        "median_translation_error_m": float(np.median([row["translation_error_m"] for row in error_rows])),
        "p95_translation_error_m": percentile([row["translation_error_m"] for row in error_rows], 95),
        "final_translation_error_m": final["translation_error_m"],
        "mean_yaw_error_degrees": float(np.mean([row["yaw_error_degrees"] for row in error_rows])),
        "median_yaw_error_degrees": float(np.median([row["yaw_error_degrees"] for row in error_rows])),
        "p95_yaw_error_degrees": percentile([row["yaw_error_degrees"] for row in error_rows], 95),
        "final_yaw_error_degrees": final["yaw_error_degrees"],
        "mean_combined_pose_error_m": float(np.mean([row["combined_pose_error_m"] for row in error_rows])),
        "median_combined_pose_error_m": float(np.median([row["combined_pose_error_m"] for row in error_rows])),
        "p95_combined_pose_error_m": percentile([row["combined_pose_error_m"] for row in error_rows], 95),
        "final_combined_pose_error_m": final["combined_pose_error_m"],
        "failure_rate": float(np.mean([row["is_failure"] for row in error_rows])),
        "lost_tracking_rate": float(np.mean([row["is_lost_tracking"] for row in error_rows])),
        "max_consecutive_lost_frames": max_consecutive([row["is_lost_tracking"] for row in error_rows]),
        "failed": int(final["is_failure"]),
        "converged": int(convergence_index is not None),
        "time_to_convergence_s": (
            error_rows[convergence_index]["replay_time_s"] if convergence_index is not None else ""
        ),
        "frames_to_convergence": convergence_index if convergence_index is not None else "",
        "post_convergence_mean_combined_pose_error_m": (
            float(np.mean([row["combined_pose_error_m"] for row in post_convergence_rows]))
            if post_convergence_rows
            else ""
        ),
        "mean_x_bias_m": float(np.mean([row["x_error_m"] for row in error_rows])),
        "mean_y_bias_m": float(np.mean([row["y_error_m"] for row in error_rows])),
        "mean_abs_x_error_m": float(np.mean([abs(row["x_error_m"]) for row in error_rows])),
        "mean_abs_y_error_m": float(np.mean([abs(row["y_error_m"]) for row in error_rows])),
        "mean_effective_particle_count": float(np.mean(ess_values)),
        "min_effective_particle_count": float(min(ess_values)),
        "resampling_rate": resample_count / len(sampled_frames),
        "recovery_event_count": recovery_event_count,
        "mean_recovery_particle_ratio": float(np.mean([row["random_particle_ratio"] for row in frame_rows])),
        "mean_render_and_score_ms": float(np.mean(render_ms_values)),
        "p95_render_and_score_ms": percentile(render_ms_values, 95),
        "mean_pf_step_ms": float(np.mean([row["pf_step_ms"] for row in frame_rows])),
        "p95_pf_step_ms": percentile([row["pf_step_ms"] for row in frame_rows], 95),
        "mean_total_frame_ms": float(np.mean(total_frame_ms_values)),
        "p95_total_frame_ms": percentile(total_frame_ms_values, 95),
        "mean_total_hz": 1000.0 / float(np.mean(total_frame_ms_values)) if total_frame_ms_values else "",
        "mean_gpu_memory_used_mb": float(np.mean(gpu_memory_used_mb_values)) if gpu_memory_used_mb_values else "",
        "max_gpu_memory_used_mb": float(max(gpu_memory_used_mb_values)) if gpu_memory_used_mb_values else "",
        "p95_gpu_memory_used_mb": percentile(gpu_memory_used_mb_values, 95) if gpu_memory_used_mb_values else "",
        "mean_gpu_memory_free_mb": float(np.mean(gpu_memory_free_mb_values)) if gpu_memory_free_mb_values else "",
        "min_gpu_memory_free_mb": float(min(gpu_memory_free_mb_values)) if gpu_memory_free_mb_values else "",
        "particle_snapshots_recorded": int(should_record_particles),
    }
    return frame_rows, summary


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix.resolve()
    matrix = load_matrix(matrix_path)
    matrix_dir = matrix_path.parent

    raw_experiment = matrix.get("experiment", {})
    run_name = args.run_name or raw_experiment.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    settings = load_turtlebot_localization_config(args.config)
    seeds = [int(seed) for seed in raw_experiment.get("seeds", DEFAULT_SEEDS)]
    frame_stride = int(raw_experiment.get("frame_stride", 5))
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    gpu_memory_poll_stride = int(raw_experiment.get("gpu_memory_poll_stride", 1))
    if gpu_memory_poll_stride <= 0:
        raise ValueError("gpu_memory_poll_stride must be positive")
    backend = str(raw_experiment.get("backend", settings.renderer.backend or "vkdiff"))
    port = int(raw_experiment.get("port", 8000))
    restart_renderer_enabled = bool(raw_experiment.get("restart_renderer", True))
    map_yaml = resolve_path(raw_experiment.get("map_yaml", "map.yaml"), base_dir=matrix_dir)
    thresholds = parse_thresholds(raw_experiment.get("thresholds"))
    particle_recording = parse_particle_recording(raw_experiment.get("record_particles"))
    splat_iterations = [int(value) for value in matrix.get("splats", {}).get("iterations", DEFAULT_SPLAT_ITERATIONS)]
    measurement = measurement_with_overrides(settings.measurement, raw_experiment.get("measurement"))
    modes = parse_modes(matrix.get("modes", {}))
    path_specs = parse_path_specs(matrix, base_dir=matrix_dir)

    free_space_sampler = FreeSpacePoseSampler.from_map_yaml(
        map_yaml,
        global_yaw_uniform=settings.initialization.global_yaw_uniform,
    )
    renderer_client = RendererServiceClient(settings.renderer)
    splats_by_csv: dict[Path, list[SplatSpec]] = {}
    manifests_by_path: dict[Path, ReplayManifest] = {}
    renderer_health_by_splat: dict[str, dict] = {}
    per_frame_rows: list[dict] = []
    per_run_rows: list[dict] = []

    for path_spec in path_specs:
        splats_by_csv.setdefault(path_spec.splat_csv, selected_splats(load_splats(path_spec.splat_csv), splat_iterations))
        manifests_by_path.setdefault(path_spec.manifest_path, ReplayManifest.load(path_spec.manifest_path))

    multi_splat_csvs = [str(csv_path) for csv_path, splats in splats_by_csv.items() if len(splats) > 1]
    if multi_splat_csvs and not restart_renderer_enabled:
        raise ValueError(
            "Matrices with multiple splats require experiment.restart_renderer: true. "
            f"Multi-splat CSVs: {', '.join(multi_splat_csvs)}"
        )

    total_runs = sum(len(splats_by_csv[path_spec.splat_csv]) * len(modes) * len(seeds) for path_spec in path_specs)
    estimated_total_frames = sum(
        math.ceil(len(manifests_by_path[path_spec.manifest_path].frames) / frame_stride)
        * len(splats_by_csv[path_spec.splat_csv])
        * len(modes)
        * len(seeds)
        for path_spec in path_specs
    )
    console = Console() if RICH_AVAILABLE else None
    progress = make_progress(enabled=not args.no_progress)
    print_study_summary(
        console=console,
        run_name=run_name,
        path_specs=path_specs,
        splats_by_csv=splats_by_csv,
        modes=modes,
        seeds=seeds,
        frame_stride=frame_stride,
        total_runs=total_runs,
        estimated_total_frames=estimated_total_frames,
    )

    run_counter = 0
    active_splat_id: str | None = None
    with progress if progress is not None else nullcontext():
        overall_task_id = progress.add_task("overall runs", total=total_runs) if progress is not None else None
        run_task_id = progress.add_task("frames", total=1, visible=False) if progress is not None else None
        for path_spec in path_specs:
            manifest = manifests_by_path[path_spec.manifest_path]
            for splat in splats_by_csv[path_spec.splat_csv]:
                if restart_renderer_enabled and active_splat_id != splat.splat_id:
                    if console is not None:
                        console.print(f"[yellow]Restarting renderer[/yellow] splat={splat.splat_id}")
                    restart_renderer(
                        splat=splat,
                        backend=backend,
                        port=port,
                        build_image=args.build_image,
                        settings=settings,
                    )
                    active_splat_id = splat.splat_id
                health = dict(renderer_client.wait_until_ready())
                renderer_health_by_splat[splat.splat_id] = health
                if console is not None:
                    console.print(
                        f"[cyan]{path_spec.scenario_id}/{path_spec.path_id}[/cyan] "
                        f"splat=[bold]{splat.splat_id}[/bold] backend={health.get('backend')} "
                        f"gaussians={health.get('gaussians')}"
                    )
                else:
                    print(
                        f"[{path_spec.scenario_id}/{path_spec.path_id}] splat={splat.splat_id} "
                        f"backend={health.get('backend')} gaussians={health.get('gaussians')}",
                        flush=True,
                    )
                for mode in modes:
                    for seed in seeds:
                        run_counter += 1
                        prior_suffix = f"__prior{mode.prior_case_index}" if mode.mode == "local" else ""
                        run_id = (
                            f"{path_spec.scenario_id}__{path_spec.path_id}__{splat.splat_id}__"
                            f"{mode.mode}{prior_suffix}__seed{seed}"
                        )
                        label = run_label(
                            scenario_id=path_spec.scenario_id,
                            path_id=path_spec.path_id,
                            splat_id=splat.splat_id,
                            mode=mode,
                            seed=seed,
                        )
                        global_rng = random.Random(seed)
                        global_sampler = lambda rng=global_rng: free_space_sampler.sample_pose(rng=rng)
                        if console is not None:
                            console.print(f"[bold]Run {run_counter}/{total_runs}[/bold] {label}")
                        else:
                            print(f"Run {run_counter}/{total_runs}: {run_id}", flush=True)
                        frame_rows, summary = evaluate_run(
                            run_id=run_id,
                            scenario_id=path_spec.scenario_id,
                            path_id=path_spec.path_id,
                            splat=splat,
                            mode=mode,
                            seed=seed,
                            manifest=manifest,
                            manifest_path=path_spec.manifest_path,
                            renderer_client=renderer_client,
                            settings=settings,
                            measurement=measurement,
                            frame_stride=frame_stride,
                            thresholds=thresholds,
                            global_sampler=global_sampler,
                            output_dir=output_dir,
                            particle_recording=particle_recording,
                            gpu_memory_poll_stride=gpu_memory_poll_stride,
                            progress=progress,
                            run_task_id=run_task_id,
                            run_label_text=label,
                        )
                        summary["renderer_backend"] = health.get("backend")
                        summary["renderer_gaussians"] = health.get("gaussians")
                        per_frame_rows.extend(frame_rows)
                        per_run_rows.append(summary)
                        write_csv(output_dir / "per_frame.csv", per_frame_rows)
                        write_csv(output_dir / "per_run_summary.csv", per_run_rows)
                        if progress is not None and overall_task_id is not None:
                            progress.update(overall_task_id, advance=1)
                        if progress is not None and run_task_id is not None:
                            progress.update(run_task_id, visible=False)
                        if console is not None:
                            console.print(
                                f"[green]done[/green] {run_id} "
                                f"mean_combined={summary['mean_combined_pose_error_m']:.3f}m "
                                f"failure_rate={summary['failure_rate']:.3f} "
                                f"mean_hz={summary['mean_total_hz']:.2f}"
                            )
                        else:
                            print(
                                f"  done mean_combined={summary['mean_combined_pose_error_m']:.3f}m "
                                f"failure_rate={summary['failure_rate']:.3f} mean_hz={summary['mean_total_hz']:.2f}",
                                flush=True,
                            )

    metadata = {
        "created_at_unix_seconds": time.time(),
        "matrix": str(matrix_path),
        "config": str(args.config),
        "run_name": run_name,
        "seeds": seeds,
        "frame_stride": frame_stride,
        "gpu_memory_poll_stride": gpu_memory_poll_stride,
        "record_particles": asdict(particle_recording),
        "map_yaml": str(map_yaml),
        "splat_iterations": splat_iterations,
        "measurement": asdict(measurement),
        "thresholds": asdict(thresholds),
        "modes": [asdict(mode) for mode in modes],
        "path_specs": [asdict(path_spec) for path_spec in path_specs],
        "renderer_health_by_splat": renderer_health_by_splat,
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Wrote experiment outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
