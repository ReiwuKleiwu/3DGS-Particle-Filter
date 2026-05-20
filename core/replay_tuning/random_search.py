"""Samples localization parameter sets at random and evaluates them on replay datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from core.config import load_turtlebot_localization_config
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.replay_tuning.evaluator import load_manifests
from core.replay_tuning.models import SearchCandidate
from core.replay_tuning.paths import RESULTS_DIR
from core.replay_tuning.search_common import evaluate_candidate_across_manifests, normalize_weights


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the random replay-tuning search tool."""
    parser = argparse.ArgumentParser(description="Random search over PF parameters using recorded replay manifests.")
    parser.add_argument("--manifest", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "random_search.json")
    parser.add_argument("--config", type=Path, default=Path("turtlebot_localization.yaml"))
    return parser.parse_args()


def sample_candidate(rng: random.Random, trial_index: int) -> SearchCandidate:
    """Samples one random parameter candidate inside the current search ranges."""
    ssim, l1, grad = normalize_weights(
        rng.uniform(0.1, 0.8),
        rng.uniform(0.05, 0.7),
        rng.uniform(0.05, 0.7),
    )
    return SearchCandidate(
        particle_count=rng.choice([128, 160, 192, 256]),
        resample_threshold_ratio=rng.uniform(0.25, 0.65),
        temperature=rng.uniform(0.01, 0.06),
        motion_noise_x=rng.uniform(0.005, 0.05),
        motion_noise_y=rng.uniform(0.005, 0.05),
        motion_noise_yaw=rng.uniform(0.005, 0.08),
        hybrid_ssim_weight=ssim,
        hybrid_l1_weight=l1,
        hybrid_gradient_weight=grad,
        lpips_top_k=rng.choice([0, 4, 8, 16]),
        lpips_weight=rng.uniform(0.0, 0.7),
        random_seed=1000 + trial_index,
        prior_sigma_x=rng.uniform(0.3, 0.8),
        prior_sigma_y=rng.uniform(0.3, 0.8),
        prior_sigma_yaw_degrees=rng.uniform(15.0, 45.0),
    )


def main() -> None:
    """Runs the random-search tuning loop and writes the aggregated results artifact."""
    args = parse_args()
    settings = load_turtlebot_localization_config(args.config)
    print(f"Loading config: {args.config}", flush=True)
    print(f"Connecting to renderer: {settings.renderer.base_url}", flush=True)
    renderer_client = RendererServiceClient(settings.renderer)
    renderer_client.wait_until_ready()
    print("Renderer is ready", flush=True)
    manifests = load_manifests(args.manifest)
    print("Loaded manifests:", flush=True)
    for manifest_path, manifest in zip(args.manifest, manifests):
        print(
            f"  - {manifest_path} | frames={len(manifest.frames)} | notes={manifest.notes!r}",
            flush=True,
        )
    print(f"Starting random search with {args.trials} trials", flush=True)
    rng = random.Random(args.seed)

    all_results = []
    for trial_index in range(args.trials):
        candidate = sample_candidate(rng, trial_index)
        print(
            f"[trial {trial_index + 1}/{args.trials}] "
            f"particles={candidate.particle_count} "
            f"temp={candidate.temperature:.4f} "
            f"resample={candidate.resample_threshold_ratio:.3f} "
            f"motion=({candidate.motion_noise_x:.3f},{candidate.motion_noise_y:.3f},{candidate.motion_noise_yaw:.3f}) "
            f"lpips_top_k={candidate.lpips_top_k} "
            f"lpips_weight={candidate.lpips_weight:.3f}",
            flush=True,
        )
        def progress(update: dict) -> None:
            update_type = update["type"]
            if update_type == "case_start":
                prior = update["prior_offset"]
                print(
                    f"    case {update['case_index'] + 1}/{update['case_count']} start "
                    f"offset=(dx={prior['dx']:.2f}, dy={prior['dy']:.2f}, dyaw={prior['dyaw_degrees']:.1f}deg)",
                    flush=True,
                )
            elif update_type == "frame_progress":
                print(
                    f"      frame {update['frame_index']}/{update['frame_count']} "
                    f"(case {update['case_index'] + 1}/{update['case_count']})",
                    flush=True,
                )
            elif update_type == "case_done":
                print(
                    f"    case {update['case_index'] + 1}/{update['case_count']} done "
                    f"translation={update['translation_error_m']:.4f} "
                    f"yaw={update['yaw_error_degrees']:.2f}deg "
                    f"failed={'yes' if update['failed'] else 'no'} "
                    f"mean_elapsed_ms={update['mean_elapsed_ms']:.1f}",
                    flush=True,
                )

        def progress_factory(manifest_path: Path):
            print(f"  evaluating manifest: {manifest_path}", flush=True)
            return progress

        result = evaluate_candidate_across_manifests(
            candidate=candidate,
            manifests=manifests,
            manifest_paths=args.manifest,
            renderer_client=renderer_client,
            settings=settings,
            frame_stride=args.frame_stride,
            progress_factory=progress_factory,
        )
        for manifest_result in result["manifests"]:
            summary = manifest_result["summary"]
            print(
                f"    objective={summary['objective']:.4f} "
                f"mean_translation={summary['mean_translation_error_m']:.4f} "
                f"mean_yaw={summary['mean_abs_yaw_error_degrees']:.2f} "
                f"failure_rate={summary['catastrophic_failure_rate']:.3f} "
                f"mean_elapsed_ms={summary['mean_elapsed_ms']:.1f}",
                flush=True,
            )
        result["trial_index"] = trial_index
        all_results.append(result)
        print(
            f"trial={trial_index:03d} "
            f"objective={result['aggregate']['objective']:.4f} "
            f"mean_translation={result['aggregate']['mean_translation_error_m']:.4f} "
            f"failure_rate={result['aggregate']['catastrophic_failure_rate']:.3f}",
            flush=True,
        )

    all_results.sort(key=lambda item: item["aggregate"]["objective"])
    payload = {
        "config": str(args.config),
        "manifests": [str(path) for path in args.manifest],
        "trials": args.trials,
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
