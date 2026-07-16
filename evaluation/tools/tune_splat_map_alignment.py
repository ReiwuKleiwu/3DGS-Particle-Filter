#!/usr/bin/env python3
"""Tune DEFAULT_SPLAT_MAP_* via VkDiff /score_batch runtime overrides."""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.models import ReplayFrame, ReplayManifest


DEFAULT_CENTER_X = -0.005
DEFAULT_CENTER_Y = 0.015
DEFAULT_CENTER_YAW = 0.008


@dataclass(frozen=True)
class AlignmentCandidate:
    x: float
    y: float
    yaw: float

    @property
    def yaw_degrees(self) -> float:
        return math.degrees(self.yaw)


@dataclass(frozen=True)
class CandidateScore:
    candidate: AlignmentCandidate
    mean_score: float
    max_score: float
    frame_scores: list[float]
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune core/rendering/config.py DEFAULT_SPLAT_MAP_X/Y/YAW by passing candidate "
            "values to the running VkDiff renderer's /score_batch endpoint."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/artifacts/datasets/small_house_default/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/artifacts/splat_map_alignment/small_house_default_vkdiff.json"))
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-index", type=int, action="append", default=None)
    parser.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    parser.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    parser.add_argument("--center-yaw-deg", type=float, default=math.degrees(DEFAULT_CENTER_YAW))
    parser.add_argument("--x-range", type=float, default=0.10)
    parser.add_argument("--y-range", type=float, default=0.10)
    parser.add_argument("--yaw-range-deg", type=float, default=5.0)
    parser.add_argument("--x-steps", type=int, default=5)
    parser.add_argument("--y-steps", type=int, default=5)
    parser.add_argument("--yaw-steps", type=int, default=5)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--shrink", type=float, default=0.35)
    parser.add_argument("--renderer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--metric", default="lpips", choices=["lpips", "hybrid", "mse", "ssim", "rgb-ssim"])
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--include-preview", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _linspace(center: float, radius: float, steps: int) -> list[float]:
    if steps <= 1:
        return [center]
    return [center - radius + (2.0 * radius * index / float(steps - 1)) for index in range(steps)]


def _candidate_grid(
    center: AlignmentCandidate,
    *,
    x_range: float,
    y_range: float,
    yaw_range: float,
    x_steps: int,
    y_steps: int,
    yaw_steps: int,
) -> list[AlignmentCandidate]:
    return [
        AlignmentCandidate(x=x, y=y, yaw=yaw)
        for x, y, yaw in itertools.product(
            _linspace(center.x, x_range, x_steps),
            _linspace(center.y, y_range, y_steps),
            _linspace(center.yaw, yaw_range, yaw_steps),
        )
    ]


def _select_frames(manifest: ReplayManifest, args: argparse.Namespace) -> list[tuple[int, ReplayFrame]]:
    if args.frame_index:
        selected = []
        for index in args.frame_index:
            if index < 0 or index >= len(manifest.frames):
                raise ValueError(f"Frame index {index} outside manifest range 0..{len(manifest.frames) - 1}")
            selected.append((index, manifest.frames[index]))
        return selected

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    indexed = [(index, frame) for index, frame in enumerate(manifest.frames) if index % args.frame_stride == 0]
    if args.frame_count <= 0 or args.frame_count >= len(indexed):
        return indexed
    if args.frame_count == 1:
        return [indexed[len(indexed) // 2]]
    step = (len(indexed) - 1) / float(args.frame_count - 1)
    return [indexed[round(i * step)] for i in range(args.frame_count)]


def _request_json(url: str, payload: dict[str, Any] | None, *, timeout: float) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _check_renderer(base_url: str, timeout: float) -> None:
    status = _request_json(f"{base_url.rstrip('/')}/health", None, timeout=timeout)
    if status.get("status") != "ok" or not status.get("renderer_loaded"):
        raise RuntimeError(f"Renderer is not ready: {status}")
    if status.get("backend") != "vkdiff":
        raise RuntimeError(f"Expected vkdiff renderer, got: {status.get('backend')}")


def _camera_payload(manifest: ReplayManifest) -> dict[str, float | int]:
    return {
        "width": manifest.camera.width,
        "height": manifest.camera.height,
        "fx": manifest.camera.fx,
        "fy": manifest.camera.fy,
        "cx": manifest.camera.cx,
        "cy": manifest.camera.cy,
    }


def _pose_payload(frame: ReplayFrame) -> dict[str, float]:
    return {"x": frame.pose.x, "y": frame.pose.y, "yaw": frame.pose.yaw}


def score_candidate(
    args: argparse.Namespace,
    manifest: ReplayManifest,
    frames: list[tuple[int, ReplayFrame]],
    candidate: AlignmentCandidate,
) -> CandidateScore:
    start = time.perf_counter()
    frame_scores: list[float] = []
    preview_written = False

    for frame_index, frame in frames:
        image_bytes = manifest.resolve_image_path(frame.image_path).read_bytes()
        payload = {
            "poses": [_pose_payload(frame)],
            "camera": _camera_payload(manifest),
            "observation_png_base64": base64.b64encode(image_bytes).decode("ascii"),
            "include_best_render_preview": bool(args.include_preview or (args.preview_dir and not preview_written)),
            "metric": args.metric,
            "lpips_net": args.lpips_net,
            "max_batch_size": args.max_batch_size,
            "splat_map_x": candidate.x,
            "splat_map_y": candidate.y,
            "splat_map_yaw": candidate.yaw,
        }
        response = _request_json(f"{args.renderer_url.rstrip('/')}/score_batch", payload, timeout=args.request_timeout)
        frame_scores.append(float(response["scores"][0]))

        if args.preview_dir and not preview_written and response.get("best_render_png_base64"):
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = args.preview_dir / (
                f"x_{candidate.x:+.4f}_y_{candidate.y:+.4f}_yaw_{candidate.yaw_degrees:+.3f}_"
                f"frame_{frame_index:06d}.png"
            )
            preview_path.write_bytes(base64.b64decode(response["best_render_png_base64"]))
            preview_written = True

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return CandidateScore(
        candidate=candidate,
        mean_score=sum(frame_scores) / max(1, len(frame_scores)),
        max_score=max(frame_scores),
        frame_scores=frame_scores,
        elapsed_ms=elapsed_ms,
    )


def _score_payload(score: CandidateScore, frame_indices: list[int]) -> dict[str, Any]:
    return {
        "DEFAULT_SPLAT_MAP_X": score.candidate.x,
        "DEFAULT_SPLAT_MAP_Y": score.candidate.y,
        "DEFAULT_SPLAT_MAP_YAW": score.candidate.yaw,
        "yaw_degrees": score.candidate.yaw_degrees,
        "mean_score": score.mean_score,
        "max_score": score.max_score,
        "elapsed_ms": score.elapsed_ms,
        "frame_scores": [
            {"frame_index": index, "score": value}
            for index, value in zip(frame_indices, score.frame_scores)
        ],
    }


def main() -> None:
    args = parse_args()
    manifest = ReplayManifest.load(args.manifest)
    frames = _select_frames(manifest, args)
    frame_indices = [index for index, _ in frames]
    center = AlignmentCandidate(args.center_x, args.center_y, math.radians(args.center_yaw_deg))
    x_range = args.x_range
    y_range = args.y_range
    yaw_range = math.radians(args.yaw_range_deg)
    best_score: CandidateScore | None = None
    all_scores: list[CandidateScore] = []

    _check_renderer(args.renderer_url, args.request_timeout)
    print(f"Scoring frames: {frame_indices}")
    print("Using runtime splat_map_x/y/yaw overrides; no renderer restart per candidate.")

    for pass_index in range(args.refinements + 1):
        candidates = _candidate_grid(
            center,
            x_range=x_range,
            y_range=y_range,
            yaw_range=yaw_range,
            x_steps=args.x_steps,
            y_steps=args.y_steps,
            yaw_steps=args.yaw_steps,
        )
        print(f"Pass {pass_index + 1}/{args.refinements + 1}: {len(candidates)} candidates")
        for candidate_index, candidate in enumerate(candidates, start=1):
            score = score_candidate(args, manifest, frames, candidate)
            all_scores.append(score)
            if best_score is None or score.mean_score < best_score.mean_score:
                best_score = score
                print(
                    f"  new best {score.mean_score:.6f} at {candidate_index}/{len(candidates)} "
                    f"x={candidate.x:.6f} y={candidate.y:.6f} yaw={candidate.yaw_degrees:.6f} deg"
                )
            elif candidate_index == len(candidates) or candidate_index % 25 == 0:
                print(f"  scored {candidate_index}/{len(candidates)} current={score.mean_score:.6f}")

        assert best_score is not None
        center = best_score.candidate
        x_range *= args.shrink
        y_range *= args.shrink
        yaw_range *= args.shrink

    assert best_score is not None
    ranked = sorted(all_scores, key=lambda item: item.mean_score)
    payload = {
        "manifest": str(args.manifest),
        "frame_indices": frame_indices,
        "metric": args.metric,
        "lpips_net": args.lpips_net,
        "best": _score_payload(best_score, frame_indices),
        "top_candidates": [_score_payload(score, frame_indices) for score in ranked[: max(1, args.top_k)]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBest constants:")
    print(f"DEFAULT_SPLAT_MAP_X = {best_score.candidate.x:.12f}")
    print(f"DEFAULT_SPLAT_MAP_Y = {best_score.candidate.y:.12f}")
    print(f"DEFAULT_SPLAT_MAP_YAW = {best_score.candidate.yaw:.12f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
