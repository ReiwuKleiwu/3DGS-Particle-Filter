#!/usr/bin/env python3
"""Step through one matrix evaluation run and save per-frame debug previews."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import load_turtlebot_localization_config
from core.particle_filter.application.step_engine import LocalizationStepEngine
from core.particle_filter.domain.recovery import AugmentedMclRecoveryTracker
from core.particle_filter.infrastructure.map import FreeSpacePoseSampler
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from evaluation.evaluator import build_observation
from evaluation.models import ReplayManifest
from evaluation.run_matrix_experiment import (
    ModeSpec,
    PathSpec,
    SplatSpec,
    build_particle_filter,
    load_matrix,
    load_splats,
    measurement_with_overrides,
    parse_modes,
    parse_path_specs,
    pose_error,
    replay_time_seconds,
    resolve_path,
    restart_renderer,
    run_label,
    selected_splats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one matrix replay run frame by frame.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("turtlebot_localization.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--path-id", default=None)
    parser.add_argument("--splat-id", default=None)
    parser.add_argument("--mode", choices=["local", "global"], default=None)
    parser.add_argument("--prior-case-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Inclusive debug frame index after stride/start selection.")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--no-restart-renderer", action="store_true")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter between frames.")
    parser.add_argument("--no-images", action="store_true", help="Write CSV/fit data without per-frame PNGs.")
    parser.add_argument("--fit-max-score", type=float, default=0.40)
    parser.add_argument("--fit-max-yaw-error-deg", type=float, default=20.0)
    parser.add_argument("--fit-max-position-error-m", type=float, default=1.0)
    parser.add_argument("--fit-max-residual-m", type=float, default=0.30)
    parser.add_argument("--fit-min-inliers", type=int, default=8)
    return parser.parse_args()


def choose_path(paths: list[PathSpec], *, scenario_id: str | None, path_id: str | None) -> PathSpec:
    matches = [
        path
        for path in paths
        if (scenario_id is None or path.scenario_id == scenario_id)
        and (path_id is None or path.path_id == path_id)
    ]
    if not matches:
        raise ValueError(f"No path matched scenario_id={scenario_id!r}, path_id={path_id!r}")
    if len(matches) > 1:
        options = ", ".join(f"{path.scenario_id}/{path.path_id}" for path in matches)
        raise ValueError(f"Path selection is ambiguous: {options}")
    return matches[0]


def choose_splat(splats: list[SplatSpec], *, splat_id: str | None) -> SplatSpec:
    matches = [splat for splat in splats if splat_id is None or splat.splat_id == splat_id]
    if not matches:
        raise ValueError(f"No splat matched splat_id={splat_id!r}")
    if len(matches) > 1:
        options = ", ".join(splat.splat_id for splat in matches)
        raise ValueError(f"Splat selection is ambiguous: {options}")
    return matches[0]


def choose_mode(modes: list[ModeSpec], *, mode_name: str | None, prior_case_index: int | None) -> ModeSpec:
    matches = [
        mode
        for mode in modes
        if (mode_name is None or mode.mode == mode_name)
        and (prior_case_index is None or mode.prior_case_index == prior_case_index)
    ]
    if not matches:
        raise ValueError(f"No mode matched mode={mode_name!r}, prior_case_index={prior_case_index!r}")
    if len(matches) > 1:
        options = ", ".join(
            f"{mode.mode}/prior{mode.prior_case_index}" if mode.mode == "local" else mode.mode
            for mode in matches
        )
        raise ValueError(f"Mode selection is ambiguous: {options}")
    return matches[0]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_label(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = position
    width, height = text_size(draw, text, font)
    draw.rectangle((x - 4, y - 3, x + width + 4, y + height + 5), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def make_comparison_image(
    *,
    observation_rgb: np.ndarray,
    best_render_png_bytes: bytes,
    title_lines: list[str],
) -> Image.Image:
    observation = Image.fromarray(observation_rgb, mode="RGB")
    if best_render_png_bytes:
        best_render = Image.open(io.BytesIO(best_render_png_bytes)).convert("RGB")
    else:
        best_render = Image.new("RGB", observation.size, color=(20, 20, 20))

    if best_render.size != observation.size:
        best_render = best_render.resize(observation.size, Image.Resampling.BILINEAR)

    diff = Image.fromarray(
        np.abs(np.asarray(observation, dtype=np.int16) - np.asarray(best_render, dtype=np.int16)).astype(np.uint8),
        mode="RGB",
    )

    font = ImageFont.load_default()
    panel_width, panel_height = observation.size
    title_height = 18 * max(1, len(title_lines)) + 10
    label_height = 22
    canvas = Image.new("RGB", (panel_width * 3, panel_height + title_height + label_height), color=(245, 245, 245))
    canvas.paste(observation, (0, title_height + label_height))
    canvas.paste(best_render, (panel_width, title_height + label_height))
    canvas.paste(diff, (panel_width * 2, title_height + label_height))

    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title_lines):
        draw.text((8, 6 + index * 18), line, fill=(20, 20, 20), font=font)
    draw_label(draw, (8, title_height + 4), "observation", font)
    draw_label(draw, (panel_width + 8, title_height + 4), "best render", font)
    draw_label(draw, (panel_width * 2 + 8, title_height + 4), "absolute diff", font)
    return canvas


def save_step_images(
    *,
    output_dir: Path,
    frame_index: int,
    observation_rgb: np.ndarray,
    best_render_png_bytes: bytes,
    title_lines: list[str],
) -> Path:
    frame_dir = output_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation_rgb, mode="RGB").save(frame_dir / "observation.png")
    if best_render_png_bytes:
        (frame_dir / "best_render.png").write_bytes(best_render_png_bytes)
    comparison = make_comparison_image(
        observation_rgb=observation_rgb,
        best_render_png_bytes=best_render_png_bytes,
        title_lines=title_lines,
    )
    comparison_path = frame_dir / "comparison.png"
    comparison.save(comparison_path)
    return comparison_path


DEBUG_CSV_COLUMNS = [
    "frame_index",
    "replay_time_s",
    "best_score",
    "render_and_score_ms",
    "total_frame_ms",
    "truth_x",
    "truth_y",
    "truth_yaw",
    "truth_yaw_degrees",
    "best_x",
    "best_y",
    "best_yaw",
    "best_yaw_degrees",
    "estimate_x",
    "estimate_y",
    "estimate_yaw",
    "estimate_yaw_degrees",
    "best_minus_truth_x",
    "best_minus_truth_y",
    "best_minus_truth_yaw_degrees",
    "estimate_translation_error_m",
    "estimate_yaw_error_degrees",
    "effective_particle_count",
    "resampled",
    "random_particle_ratio",
    "random_particle_count",
    "roughening_particle_count",
    "fit_residual_m",
]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix_2d(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def filter_fit_rows(
    rows: list[dict],
    *,
    max_score: float,
    max_yaw_error_deg: float,
    max_position_error_m: float,
) -> list[dict]:
    inliers = []
    for row in rows:
        position_error = math.hypot(row["best_minus_truth_x"], row["best_minus_truth_y"])
        yaw_error = abs(row["best_minus_truth_yaw_degrees"])
        if (
            row["best_score"] <= max_score
            and yaw_error <= max_yaw_error_deg
            and position_error <= max_position_error_m
        ):
            inliers.append(row)
    return inliers


def fit_similarity_truth_to_best(
    rows: list[dict],
    *,
    total_frame_count: int,
    min_inliers: int,
    fit_filters: dict,
) -> dict:
    if len(rows) < 2:
        return {
            "plausible": False,
            "reason": "Need at least two frames to fit scale/translation.",
            "total_frame_count": total_frame_count,
            "inlier_count": len(rows),
            "fit_filters": fit_filters,
        }

    truth = np.array([[row["truth_x"], row["truth_y"]] for row in rows], dtype=np.float64)
    best = np.array([[row["best_x"], row["best_y"]] for row in rows], dtype=np.float64)
    truth_mean = truth.mean(axis=0)
    best_mean = best.mean(axis=0)
    truth_centered = truth - truth_mean
    best_centered = best - best_mean
    truth_variance = float(np.mean(np.sum(truth_centered * truth_centered, axis=1)))
    truth_span = float(np.max(np.linalg.norm(truth_centered, axis=1)))
    if truth_variance <= 1e-12:
        return {
            "plausible": False,
            "reason": "Truth positions have too little spread to fit scale.",
            "total_frame_count": total_frame_count,
            "inlier_count": len(rows),
            "fit_filters": fit_filters,
            "truth_span_m": truth_span,
        }

    covariance = (best_centered.T @ truth_centered) / len(rows)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    orientation = np.eye(2, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        orientation[1, 1] = -1.0
    rotation = u_matrix @ orientation @ vt_matrix
    scale = float(np.sum(singular_values * np.diag(orientation)) / truth_variance)

    yaw_deltas = np.array([wrap_angle(row["best_yaw"] - row["truth_yaw"]) for row in rows], dtype=np.float64)
    yaw_delta = float(math.atan2(float(np.mean(np.sin(yaw_deltas))), float(np.mean(np.cos(yaw_deltas)))))
    yaw_delta_std = float(np.sqrt(max(0.0, 1.0 - math.hypot(float(np.mean(np.cos(yaw_deltas))), float(np.mean(np.sin(yaw_deltas)))))))

    # Prefer heading deltas for yaw. Recompute translation with that yaw and the fitted scale.
    rotation_from_yaw = rotation_matrix_2d(yaw_delta)
    translation = best_mean - scale * (rotation_from_yaw @ truth_mean)
    predicted = (scale * (rotation_from_yaw @ truth.T)).T + translation
    residuals = np.linalg.norm(best - predicted, axis=1)
    rms_residual = float(np.sqrt(np.mean(residuals * residuals)))
    max_residual = float(np.max(residuals))

    position_fit_yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
    plausible = (
        len(rows) >= min_inliers
        and truth_span >= 0.10
        and 0.25 <= scale <= 2.5
        and rms_residual <= 0.25
        and math.degrees(yaw_delta_std) <= 20.0
    )
    reasons = []
    if len(rows) < min_inliers:
        reasons.append(f"fewer than {min_inliers} inlier frames")
    if truth_span < 0.10:
        reasons.append(f"truth path spread is small ({truth_span:.3f} m)")
    if not 0.25 <= scale <= 2.5:
        reasons.append(f"scale is outside sanity bounds ({scale:.3f})")
    if rms_residual > 0.25:
        reasons.append(f"RMS residual is high ({rms_residual:.3f} m)")
    if math.degrees(yaw_delta_std) > 20.0:
        reasons.append(f"yaw deltas are noisy ({math.degrees(yaw_delta_std):.1f} deg circular std proxy)")

    return {
        "plausible": plausible,
        "reason": "ok" if plausible else "; ".join(reasons),
        "total_frame_count": total_frame_count,
        "inlier_count": len(rows),
        "fit_filters": fit_filters,
        "truth_span_m": truth_span,
        "fit_truth_to_best": {
            "translation_x": float(translation[0]),
            "translation_y": float(translation[1]),
            "scale": scale,
            "yaw_degrees_from_headings": math.degrees(yaw_delta),
            "yaw_degrees_from_positions": math.degrees(position_fit_yaw),
            "rms_residual_m": rms_residual,
            "max_residual_m": max_residual,
            "yaw_delta_std_proxy_degrees": math.degrees(yaw_delta_std),
        },
    }


def residual_trim_fit_rows(rows: list[dict], fit: dict, *, max_residual_m: float) -> list[dict]:
    if "fit_truth_to_best" not in fit:
        return rows
    fit_values = fit["fit_truth_to_best"]
    scale = float(fit_values["scale"])
    yaw = math.radians(float(fit_values["yaw_degrees_from_headings"]))
    translation = np.array(
        [float(fit_values["translation_x"]), float(fit_values["translation_y"])],
        dtype=np.float64,
    )
    rotation = rotation_matrix_2d(yaw)
    trimmed = []
    for row in rows:
        truth = np.array([row["truth_x"], row["truth_y"]], dtype=np.float64)
        best = np.array([row["best_x"], row["best_y"]], dtype=np.float64)
        predicted = scale * (rotation @ truth) + translation
        residual = float(np.linalg.norm(best - predicted))
        next_row = dict(row)
        next_row["fit_residual_m"] = residual
        if residual <= max_residual_m:
            trimmed.append(next_row)
    return trimmed


def compose_renderer_alignment(settings, fit: dict) -> dict:
    if not fit.get("plausible"):
        return {}

    current_scale = float(settings.renderer.splat_map_scale)
    current_yaw = math.radians(float(settings.renderer.splat_map_yaw_degrees))
    current_translation = np.array(
        [float(settings.renderer.splat_map_x), float(settings.renderer.splat_map_y)],
        dtype=np.float64,
    )

    fit_values = fit["fit_truth_to_best"]
    correction_translation = np.array(
        [fit_values["translation_x"], fit_values["translation_y"]],
        dtype=np.float64,
    )
    correction_scale = float(fit_values["scale"])
    correction_yaw = math.radians(float(fit_values["yaw_degrees_from_headings"]))

    current_linear = current_scale * rotation_matrix_2d(current_yaw)
    next_translation = current_translation + current_linear @ correction_translation
    next_scale = current_scale * correction_scale
    next_yaw = wrap_angle(current_yaw + correction_yaw)

    return {
        "splat_map_x": float(next_translation[0]),
        "splat_map_y": float(next_translation[1]),
        "splat_map_scale": float(next_scale),
        "splat_map_yaw_degrees": float(math.degrees(next_yaw)),
    }


def main() -> None:
    args = parse_args()
    if args.frame_stride is not None and args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("--end-frame must be greater than or equal to --start-frame")

    matrix_path = args.matrix.resolve()
    matrix = load_matrix(matrix_path)
    matrix_dir = matrix_path.parent
    raw_experiment = matrix.get("experiment", {})
    settings = load_turtlebot_localization_config(args.config)
    seed = int(args.seed if args.seed is not None else (raw_experiment.get("seeds") or [1001])[0])
    frame_stride = int(args.frame_stride if args.frame_stride is not None else raw_experiment.get("frame_stride", 1))
    backend = str(raw_experiment.get("backend", settings.renderer.backend or "vkdiff"))
    port = int(raw_experiment.get("port", 8000))
    map_yaml = resolve_path(raw_experiment.get("map_yaml", "map.yaml"), base_dir=matrix_dir)
    measurement = measurement_with_overrides(settings.measurement, raw_experiment.get("measurement"))

    path = choose_path(
        parse_path_specs(matrix, base_dir=matrix_dir),
        scenario_id=args.scenario_id,
        path_id=args.path_id,
    )
    splat_iterations = [int(value) for value in matrix.get("splats", {}).get("iterations", [])]
    splats = selected_splats(load_splats(path.splat_csv), splat_iterations)
    splat = choose_splat(splats, splat_id=args.splat_id)
    mode = choose_mode(
        parse_modes(matrix.get("modes", {})),
        mode_name=args.mode,
        prior_case_index=args.prior_case_index,
    )

    manifest = ReplayManifest.load(path.manifest_path)
    sampled_frames = manifest.frames[::frame_stride]
    sampled_frames = sampled_frames[args.start_frame :]
    if args.end_frame is not None:
        sampled_frames = sampled_frames[: args.end_frame - args.start_frame + 1]
    if args.max_frames is not None:
        sampled_frames = sampled_frames[: args.max_frames]
    if not sampled_frames:
        raise ValueError("No frames selected for debug run")

    run_id = (
        f"{path.scenario_id}__{path.path_id}__{splat.splat_id}__{mode.mode}"
        f"{f'__prior{mode.prior_case_index}' if mode.mode == 'local' else ''}__seed{seed}"
    )
    output_dir = args.output_dir or (REPO_ROOT / "evaluation" / "artifacts" / "debug" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_restart_renderer:
        restart_renderer(splat=splat, backend=backend, port=port, build_image=args.build_image, settings=settings)

    renderer_client = RendererServiceClient(settings.renderer)
    renderer_status = renderer_client.wait_until_ready()
    print(
        "Debug run | "
        f"{run_label(scenario_id=path.scenario_id, path_id=path.path_id, splat_id=splat.splat_id, mode=mode, seed=seed)}"
    )
    print(f"Renderer: {renderer_status.get('backend')} | splat={renderer_status.get('splat_path')}")
    print(f"Output: {output_dir}")

    rng = random.Random(seed)
    free_space_sampler = FreeSpacePoseSampler.from_map_yaml(
        map_yaml,
        global_yaw_uniform=settings.initialization.global_yaw_uniform,
    )
    global_sampler = lambda: free_space_sampler.sample_pose(rng=rng)
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
    debug_rows: list[dict] = []
    csv_path = output_dir / "debug_frames.csv"

    for debug_index, frame in enumerate(sampled_frames):
        frame_index = args.start_frame + debug_index
        observation = build_observation(manifest, frame, frame_index, settings.camera_override)
        start = time.perf_counter()
        step_result = step_engine.run_step(
            particle_filter=particle_filter,
            observation=observation,
            previous_odometry_pose=previous_odom_pose,
            recovery_tracker=recovery_tracker,
            random_pose_sampler=global_sampler,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        previous_odom_pose = step_result.previous_odometry_pose

        errors = pose_error(step_result.estimated_pose, frame.pose)
        replay_time_s = replay_time_seconds(frame, first_stamp_ns, frame_index)
        best_pose = step_result.best_particle_pose
        best_minus_truth_yaw = wrap_angle(best_pose.yaw - frame.pose.yaw)
        debug_row = {
            "frame_index": frame_index,
            "replay_time_s": replay_time_s,
            "best_score": step_result.best_score,
            "render_and_score_ms": step_result.score_result.elapsed_milliseconds,
            "total_frame_ms": elapsed_ms,
            "truth_x": frame.pose.x,
            "truth_y": frame.pose.y,
            "truth_yaw": frame.pose.yaw,
            "truth_yaw_degrees": math.degrees(frame.pose.yaw),
            "best_x": best_pose.x,
            "best_y": best_pose.y,
            "best_yaw": best_pose.yaw,
            "best_yaw_degrees": math.degrees(best_pose.yaw),
            "estimate_x": step_result.estimated_pose.x,
            "estimate_y": step_result.estimated_pose.y,
            "estimate_yaw": step_result.estimated_pose.yaw,
            "estimate_yaw_degrees": math.degrees(step_result.estimated_pose.yaw),
            "best_minus_truth_x": best_pose.x - frame.pose.x,
            "best_minus_truth_y": best_pose.y - frame.pose.y,
            "best_minus_truth_yaw_degrees": math.degrees(best_minus_truth_yaw),
            "estimate_translation_error_m": errors["translation_error_m"],
            "estimate_yaw_error_degrees": errors["yaw_error_degrees"],
            "effective_particle_count": step_result.effective_particle_count,
            "resampled": int(step_result.resampled),
            "random_particle_ratio": step_result.random_particle_ratio,
            "random_particle_count": step_result.random_particle_count,
            "roughening_particle_count": step_result.roughening_particle_count,
        }
        debug_rows.append(debug_row)
        title_lines = [
            f"frame={frame_index} replay={replay_time_s:.2f}s best_score={step_result.best_score:.4f}",
            (
                f"truth=({frame.pose.x:.2f}, {frame.pose.y:.2f}, {np.degrees(frame.pose.yaw):.1f}deg) "
                f"estimate=({step_result.estimated_pose.x:.2f}, {step_result.estimated_pose.y:.2f}, "
                f"{np.degrees(step_result.estimated_pose.yaw):.1f}deg)"
            ),
            (
                f"best_particle=({best_pose.x:.2f}, {best_pose.y:.2f}, {np.degrees(best_pose.yaw):.1f}deg) "
                f"err={errors['translation_error_m']:.3f}m/{errors['yaw_error_degrees']:.1f}deg "
                f"ess={step_result.effective_particle_count:.1f} resampled={int(step_result.resampled)}"
            ),
        ]
        comparison_path = None
        if not args.no_images:
            comparison_path = save_step_images(
                output_dir=output_dir,
                frame_index=frame_index,
                observation_rgb=observation.image_rgb,
                best_render_png_bytes=step_result.score_result.best_render_png_bytes,
                title_lines=title_lines,
            )

        print(
            f"[{debug_index + 1}/{len(sampled_frames)}] frame={frame_index} "
            f"score={step_result.best_score:.4f} "
            f"err={errors['translation_error_m']:.3f}m/{errors['yaw_error_degrees']:.1f}deg "
            f"best=({best_pose.x:.3f}, {best_pose.y:.3f}, {np.degrees(best_pose.yaw):.1f}deg) "
            f"estimate=({step_result.estimated_pose.x:.3f}, {step_result.estimated_pose.y:.3f}, "
            f"{np.degrees(step_result.estimated_pose.yaw):.1f}deg) "
            f"time={elapsed_ms:.1f}ms"
        )
        if comparison_path is not None:
            print(f"  {comparison_path}")

        if not args.no_pause and debug_index + 1 < len(sampled_frames):
            response = input("Enter: next frame | q: quit > ").strip().lower()
            if response in {"q", "quit", "exit"}:
                break

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(debug_rows)
    print(f"Wrote debug CSV: {csv_path}")

    fit_filters = {
        "max_score": args.fit_max_score,
        "max_yaw_error_deg": args.fit_max_yaw_error_deg,
        "max_position_error_m": args.fit_max_position_error_m,
        "max_residual_m": args.fit_max_residual_m,
    }
    initial_fit_rows = filter_fit_rows(
        debug_rows,
        max_score=args.fit_max_score,
        max_yaw_error_deg=args.fit_max_yaw_error_deg,
        max_position_error_m=args.fit_max_position_error_m,
    )
    initial_fit = fit_similarity_truth_to_best(
        initial_fit_rows,
        total_frame_count=len(debug_rows),
        min_inliers=args.fit_min_inliers,
        fit_filters=fit_filters,
    )
    fit_rows = residual_trim_fit_rows(initial_fit_rows, initial_fit, max_residual_m=args.fit_max_residual_m)
    inlier_csv_path = output_dir / "debug_fit_inliers.csv"
    with inlier_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(fit_rows)
    print(
        f"Wrote fit inlier CSV: {inlier_csv_path} "
        f"({len(fit_rows)}/{len(debug_rows)} frames, initial={len(initial_fit_rows)})"
    )

    fit = fit_similarity_truth_to_best(
        fit_rows,
        total_frame_count=len(debug_rows),
        min_inliers=args.fit_min_inliers,
        fit_filters=fit_filters,
    )
    recommendation = compose_renderer_alignment(settings, fit)
    fit_payload = {
        "current_renderer_alignment": {
            "splat_map_x": settings.renderer.splat_map_x,
            "splat_map_y": settings.renderer.splat_map_y,
            "splat_map_scale": settings.renderer.splat_map_scale,
            "splat_map_yaw_degrees": settings.renderer.splat_map_yaw_degrees,
        },
        "initial_fit_before_residual_trim": initial_fit,
        "fit": fit,
        "recommended_renderer_alignment": recommendation,
    }
    fit_path = output_dir / "transform_fit.json"
    fit_path.write_text(json.dumps(fit_payload, indent=2), encoding="utf-8")
    print(f"Wrote transform fit: {fit_path}")
    if recommendation:
        print("Recommended renderer config:")
        print(f"  splat_map_x: {recommendation['splat_map_x']:.6f}")
        print(f"  splat_map_y: {recommendation['splat_map_y']:.6f}")
        print(f"  splat_map_scale: {recommendation['splat_map_scale']:.6f}")
        print(f"  splat_map_yaw_degrees: {recommendation['splat_map_yaw_degrees']:.6f}")
    else:
        print(f"Transform fit not plausible: {fit.get('reason')}")


if __name__ == "__main__":
    main()
