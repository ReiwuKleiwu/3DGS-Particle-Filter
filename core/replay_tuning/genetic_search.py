"""Evolves localization parameter sets with a genetic search over replay datasets."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import load_turtlebot_localization_config
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.replay_tuning.evaluator import load_manifests
from core.replay_tuning.models import SearchCandidate
from core.replay_tuning.paths import RESULTS_DIR
from core.replay_tuning.search_common import evaluate_candidate_across_manifests, normalize_weights


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the genetic replay-tuning search tool."""
    parser = argparse.ArgumentParser(description="Genetic search over PF parameters using recorded replay manifests.")
    parser.add_argument("--manifest", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "genetic_search.json")
    parser.add_argument("--config", type=Path, default=Path("turtlebot_localization.yaml"))
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def metric_aware_candidate(
    *,
    rng: random.Random,
    trial_seed: int,
    metric_name: str,
    settings,
) -> SearchCandidate:
    """Creates an initial candidate while respecting the active scoring metric mode."""
    if metric_name == "lpips":
        ssim = settings.measurement.hybrid_ssim_weight
        l1 = settings.measurement.hybrid_l1_weight
        grad = settings.measurement.hybrid_gradient_weight
        lpips_top_k = settings.measurement.lpips_top_k
        lpips_weight = settings.measurement.lpips_weight
    else:
        ssim, l1, grad = normalize_weights(
            rng.uniform(0.1, 0.8),
            rng.uniform(0.05, 0.7),
            rng.uniform(0.05, 0.7),
        )
        lpips_top_k = rng.choice([0, 4, 8, 16])
        lpips_weight = rng.uniform(0.0, 0.7)

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
        lpips_top_k=lpips_top_k,
        lpips_weight=lpips_weight,
        random_seed=trial_seed,
        prior_sigma_x=rng.uniform(0.3, 0.8),
        prior_sigma_y=rng.uniform(0.3, 0.8),
        prior_sigma_yaw_degrees=rng.uniform(15.0, 45.0),
    )


def blend(a: float, b: float, rng: random.Random, low: float, high: float) -> float:
    alpha = rng.random()
    return clamp(alpha * a + (1.0 - alpha) * b, low, high)


def crossover(
    parent_a: SearchCandidate,
    parent_b: SearchCandidate,
    *,
    rng: random.Random,
    metric_name: str,
    child_seed: int,
    settings,
) -> SearchCandidate:
    """Combines two parent candidates into one child candidate inside valid parameter bounds."""
    if metric_name == "lpips":
        ssim = settings.measurement.hybrid_ssim_weight
        l1 = settings.measurement.hybrid_l1_weight
        grad = settings.measurement.hybrid_gradient_weight
        lpips_top_k = settings.measurement.lpips_top_k
        lpips_weight = settings.measurement.lpips_weight
    else:
        ssim, l1, grad = normalize_weights(
            blend(parent_a.hybrid_ssim_weight, parent_b.hybrid_ssim_weight, rng, 0.0, 1.0),
            blend(parent_a.hybrid_l1_weight, parent_b.hybrid_l1_weight, rng, 0.0, 1.0),
            blend(parent_a.hybrid_gradient_weight, parent_b.hybrid_gradient_weight, rng, 0.0, 1.0),
        )
        lpips_top_k = rng.choice([parent_a.lpips_top_k, parent_b.lpips_top_k])
        lpips_weight = blend(parent_a.lpips_weight, parent_b.lpips_weight, rng, 0.0, 0.7)

    return SearchCandidate(
        particle_count=rng.choice([parent_a.particle_count, parent_b.particle_count]),
        resample_threshold_ratio=blend(parent_a.resample_threshold_ratio, parent_b.resample_threshold_ratio, rng, 0.25, 0.65),
        temperature=blend(parent_a.temperature, parent_b.temperature, rng, 0.01, 0.06),
        motion_noise_x=blend(parent_a.motion_noise_x, parent_b.motion_noise_x, rng, 0.005, 0.05),
        motion_noise_y=blend(parent_a.motion_noise_y, parent_b.motion_noise_y, rng, 0.005, 0.05),
        motion_noise_yaw=blend(parent_a.motion_noise_yaw, parent_b.motion_noise_yaw, rng, 0.005, 0.08),
        hybrid_ssim_weight=ssim,
        hybrid_l1_weight=l1,
        hybrid_gradient_weight=grad,
        lpips_top_k=lpips_top_k,
        lpips_weight=lpips_weight,
        random_seed=child_seed,
        prior_sigma_x=blend(parent_a.prior_sigma_x, parent_b.prior_sigma_x, rng, 0.3, 0.8),
        prior_sigma_y=blend(parent_a.prior_sigma_y, parent_b.prior_sigma_y, rng, 0.3, 0.8),
        prior_sigma_yaw_degrees=blend(parent_a.prior_sigma_yaw_degrees, parent_b.prior_sigma_yaw_degrees, rng, 15.0, 45.0),
    )


def mutate(
    candidate: SearchCandidate,
    *,
    rng: random.Random,
    metric_name: str,
    mutation_rate: float,
    child_seed: int,
    settings,
) -> SearchCandidate:
    """Applies random parameter perturbations to a candidate for the next generation."""
    data = asdict(candidate)

    def maybe(value, sampler):
        return sampler() if rng.random() < mutation_rate else value

    data["particle_count"] = maybe(data["particle_count"], lambda: rng.choice([128, 160, 192, 256]))
    data["resample_threshold_ratio"] = maybe(data["resample_threshold_ratio"], lambda: rng.uniform(0.25, 0.65))
    data["temperature"] = maybe(data["temperature"], lambda: rng.uniform(0.01, 0.06))
    data["motion_noise_x"] = maybe(data["motion_noise_x"], lambda: rng.uniform(0.005, 0.05))
    data["motion_noise_y"] = maybe(data["motion_noise_y"], lambda: rng.uniform(0.005, 0.05))
    data["motion_noise_yaw"] = maybe(data["motion_noise_yaw"], lambda: rng.uniform(0.005, 0.08))
    data["prior_sigma_x"] = maybe(data["prior_sigma_x"], lambda: rng.uniform(0.3, 0.8))
    data["prior_sigma_y"] = maybe(data["prior_sigma_y"], lambda: rng.uniform(0.3, 0.8))
    data["prior_sigma_yaw_degrees"] = maybe(data["prior_sigma_yaw_degrees"], lambda: rng.uniform(15.0, 45.0))

    if metric_name != "lpips":
        ssim = maybe(data["hybrid_ssim_weight"], lambda: rng.uniform(0.1, 0.8))
        l1 = maybe(data["hybrid_l1_weight"], lambda: rng.uniform(0.05, 0.7))
        grad = maybe(data["hybrid_gradient_weight"], lambda: rng.uniform(0.05, 0.7))
        ssim, l1, grad = normalize_weights(ssim, l1, grad)
        data["hybrid_ssim_weight"] = ssim
        data["hybrid_l1_weight"] = l1
        data["hybrid_gradient_weight"] = grad
        data["lpips_top_k"] = maybe(data["lpips_top_k"], lambda: rng.choice([0, 4, 8, 16]))
        data["lpips_weight"] = maybe(data["lpips_weight"], lambda: rng.uniform(0.0, 0.7))
    else:
        data["hybrid_ssim_weight"] = settings.measurement.hybrid_ssim_weight
        data["hybrid_l1_weight"] = settings.measurement.hybrid_l1_weight
        data["hybrid_gradient_weight"] = settings.measurement.hybrid_gradient_weight
        data["lpips_top_k"] = settings.measurement.lpips_top_k
        data["lpips_weight"] = settings.measurement.lpips_weight

    data["random_seed"] = child_seed
    return SearchCandidate(**data)


def tournament_select(scored_population: list[dict], rng: random.Random, tournament_size: int = 3) -> SearchCandidate:
    """Selects one parent candidate using tournament selection on aggregated objective score."""
    contenders = rng.sample(scored_population, k=min(tournament_size, len(scored_population)))
    contenders.sort(key=lambda item: item["aggregate"]["objective"])
    return SearchCandidate(**contenders[0]["candidate"])


class ObjectivePlotter:
    def __init__(self, output_path: Path) -> None:
        """Initializes plot output paths and plotting state for search progress artifacts."""
        self._output_path = output_path
        self._objective_png_path = output_path.with_name(output_path.stem + "_objective").with_suffix(".png")
        self._objective_pdf_path = output_path.with_name(output_path.stem + "_objective").with_suffix(".pdf")
        self._error_png_path = output_path.with_name(output_path.stem + "_best_error").with_suffix(".png")
        self._error_pdf_path = output_path.with_name(output_path.stem + "_best_error").with_suffix(".pdf")
        self._elapsed_seconds: list[float] = []
        self._objectives: list[float] = []
        self._running_best: list[float] = []
        self._generation_indices: list[int] = []
        self._labels: list[str] = []
        self._generation_best_elapsed_seconds: list[float] = []
        self._generation_best_translation_errors: list[float] = []
        self._generation_best_yaw_errors: list[float] = []
        self._generation_best_failure_rates: list[float] = []

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.dpi": 160,
                "savefig.dpi": 300,
            }
        )

    def add_point(self, *, elapsed_seconds: float, objective: float, generation_index: int, label: str) -> None:
        """Records one evaluated population member for the running progress plots."""
        self._elapsed_seconds.append(elapsed_seconds)
        self._objectives.append(objective)
        best_so_far = objective if not self._running_best else min(self._running_best[-1], objective)
        self._running_best.append(best_so_far)
        self._generation_indices.append(generation_index)
        self._labels.append(label)

    def add_generation_best(
        self,
        *,
        elapsed_seconds: float,
        translation_error_m: float,
        yaw_error_degrees: float,
        failure_rate: float,
    ) -> None:
        """Records the best member of one generation for the error-over-time plot."""
        self._generation_best_elapsed_seconds.append(elapsed_seconds)
        self._generation_best_translation_errors.append(translation_error_m)
        self._generation_best_yaw_errors.append(yaw_error_degrees)
        self._generation_best_failure_rates.append(failure_rate)

    def save(self) -> None:
        """Writes the currently accumulated progress plots to disk."""
        if not self._elapsed_seconds:
            return

        self._save_objective_plot()
        self._save_error_plot()

    def _save_objective_plot(self) -> None:
        fig, ax_obj = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        generation_ids = sorted(set(self._generation_indices))
        color_map = plt.cm.viridis
        for generation_index in generation_ids:
            xs = [x for x, g in zip(self._elapsed_seconds, self._generation_indices) if g == generation_index]
            ys = [y for y, g in zip(self._objectives, self._generation_indices) if g == generation_index]
            color = color_map(generation_index / max(len(generation_ids) - 1, 1))
            ax_obj.scatter(
                xs,
                ys,
                s=24,
                alpha=0.75,
                color=color,
                edgecolors="none",
                label=f"Generation {generation_index + 1}",
            )

        ax_obj.plot(
            self._elapsed_seconds,
            self._running_best,
            color="#111111",
            linewidth=2.2,
            label="Running Best Objective",
            zorder=3,
        )
        ax_obj.plot(
            self._elapsed_seconds,
            self._objectives,
            color="#4C78A8",
            linewidth=1.2,
            alpha=0.35,
            zorder=2,
        )

        ax_obj.set_title("Genetic Search Progress")
        ax_obj.set_ylabel("Objective")
        ax_obj.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax_obj.spines["top"].set_visible(False)
        ax_obj.spines["right"].set_visible(False)
        handles, labels = ax_obj.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_obj.legend(by_label.values(), by_label.keys(), frameon=False, loc="upper right")
        self._objective_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self._objective_png_path, bbox_inches="tight")
        fig.savefig(self._objective_pdf_path, bbox_inches="tight")
        plt.close(fig)

    def _save_error_plot(self) -> None:
        if not self._generation_best_elapsed_seconds:
            return

        fig, ax_err = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        if self._generation_best_elapsed_seconds:
            ax_err.plot(
                self._generation_best_elapsed_seconds,
                self._generation_best_translation_errors,
                color="#D62728",
                linewidth=2.0,
                marker="o",
                markersize=4,
                label="Best Member Translation Error",
            )
            ax_err.set_ylabel("Translation Error [m]", color="#D62728")
            ax_err.tick_params(axis="y", colors="#D62728")

            ax_err_right = ax_err.twinx()
            ax_err_right.plot(
                self._generation_best_elapsed_seconds,
                self._generation_best_yaw_errors,
                color="#2CA02C",
                linewidth=1.8,
                marker="s",
                markersize=3.5,
                label="Best Member Yaw Error",
            )
            ax_err_right.set_ylabel("Yaw Error [deg]", color="#2CA02C")
            ax_err_right.tick_params(axis="y", colors="#2CA02C")

            if any(rate > 0.0 for rate in self._generation_best_failure_rates):
                for elapsed, rate in zip(self._generation_best_elapsed_seconds, self._generation_best_failure_rates):
                    if rate > 0.0:
                        ax_err.axvspan(elapsed - 0.05, elapsed + 0.05, color="#9467BD", alpha=0.15)

        ax_err.set_xlabel("Elapsed Time [s]")
        ax_err.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.9)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)
        self._error_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self._error_png_path, bbox_inches="tight")
        fig.savefig(self._error_pdf_path, bbox_inches="tight")
        plt.close(fig)

    @property
    def objective_png_path(self) -> Path:
        return self._objective_png_path

    @property
    def objective_pdf_path(self) -> Path:
        return self._objective_pdf_path

    @property
    def error_png_path(self) -> Path:
        return self._error_png_path

    @property
    def error_pdf_path(self) -> Path:
        return self._error_pdf_path


def evaluate_population(population, manifests, manifest_paths, renderer_client, settings, *, generation_index: int, plotter: ObjectivePlotter, search_start: float, frame_stride: int):
    """Evaluates one full population and updates progress plots as members complete."""
    scored = []
    for member_index, candidate in enumerate(population):
        print(
            f"  member {member_index + 1}/{len(population)} "
            f"particles={candidate.particle_count} temp={candidate.temperature:.4f} "
            f"resample={candidate.resample_threshold_ratio:.3f}",
            flush=True,
        )
        def progress_factory(manifest_path: Path):
            print(f"    manifest: {manifest_path}", flush=True)

            def progress(update: dict) -> None:
                update_type = update["type"]
                if update_type == "case_start":
                    prior = update["prior_offset"]
                    print(
                        f"      case {update['case_index'] + 1}/{update['case_count']} start "
                        f"offset=(dx={prior['dx']:.2f}, dy={prior['dy']:.2f}, dyaw={prior['dyaw_degrees']:.1f}deg)",
                        flush=True,
                    )
                elif update_type == "frame_progress":
                    print(
                        f"        frame {update['frame_index']}/{update['frame_count']}",
                        flush=True,
                    )
                elif update_type == "case_done":
                    print(
                        f"      case {update['case_index'] + 1}/{update['case_count']} done "
                        f"translation={update['translation_error_m']:.4f} "
                        f"yaw={update['yaw_error_degrees']:.2f}deg "
                        f"failed={'yes' if update['failed'] else 'no'} "
                        f"mean_elapsed_ms={update['mean_elapsed_ms']:.1f}",
                        flush=True,
                    )
            return progress

        scored_member = evaluate_candidate_across_manifests(
            candidate=candidate,
            manifests=manifests,
            manifest_paths=manifest_paths,
            renderer_client=renderer_client,
            settings=settings,
            frame_stride=frame_stride,
            progress_factory=progress_factory,
        )
        print(
            f"  member done objective={scored_member['aggregate']['objective']:.4f} "
            f"translation={scored_member['aggregate']['mean_translation_error_m']:.4f} "
            f"failure={scored_member['aggregate']['catastrophic_failure_rate']:.3f}",
            flush=True,
        )
        plotter.add_point(
            elapsed_seconds=time.perf_counter() - search_start,
            objective=scored_member["aggregate"]["objective"],
            generation_index=generation_index,
            label=f"g{generation_index + 1}_m{member_index + 1}",
        )
        plotter.save()
        scored.append(scored_member)
    scored.sort(key=lambda item: item["aggregate"]["objective"])
    return scored


def main() -> None:
    """Runs the genetic-search tuning loop and writes metrics plus plot artifacts."""
    args = parse_args()
    settings = load_turtlebot_localization_config(args.config)
    metric_name = settings.measurement.metric_name
    print(f"Loading config: {args.config}", flush=True)
    print(f"Active measurement metric: {metric_name}", flush=True)
    print(f"Connecting to renderer: {settings.renderer.base_url}", flush=True)
    renderer_client = RendererServiceClient(settings.renderer)
    renderer_client.wait_until_ready()
    print("Renderer is ready", flush=True)
    manifests = load_manifests(args.manifest)
    print("Loaded manifests:", flush=True)
    for manifest_path, manifest in zip(args.manifest, manifests):
        print(f"  - {manifest_path} | frames={len(manifest.frames)} | notes={manifest.notes!r}", flush=True)

    rng = random.Random(args.seed)
    search_start = time.perf_counter()
    plotter = ObjectivePlotter(args.output)
    next_seed = 1000
    population = [
        metric_aware_candidate(
            rng=rng,
            trial_seed=next_seed + index,
            metric_name=metric_name,
            settings=settings,
        )
        for index in range(args.population_size)
    ]
    next_seed += args.population_size

    history = []
    best_overall = None
    for generation_index in range(args.generations):
        print(
            f"[generation {generation_index + 1}/{args.generations}] population={len(population)}",
            flush=True,
        )
        scored_population = evaluate_population(
            population,
            manifests,
            args.manifest,
            renderer_client,
            settings,
            generation_index=generation_index,
            plotter=plotter,
            search_start=search_start,
            frame_stride=args.frame_stride,
        )
        best_generation = scored_population[0]
        plotter.add_generation_best(
            elapsed_seconds=time.perf_counter() - search_start,
            translation_error_m=best_generation["aggregate"]["mean_translation_error_m"],
            yaw_error_degrees=best_generation["aggregate"]["mean_abs_yaw_error_degrees"],
            failure_rate=best_generation["aggregate"]["catastrophic_failure_rate"],
        )
        plotter.save()
        history.append(
            {
                "generation": generation_index,
                "best": best_generation,
                "population": scored_population,
            }
        )
        if best_overall is None or best_generation["aggregate"]["objective"] < best_overall["aggregate"]["objective"]:
            best_overall = best_generation
        print(
            f"[generation {generation_index + 1}] best objective={best_generation['aggregate']['objective']:.4f} "
            f"translation={best_generation['aggregate']['mean_translation_error_m']:.4f} "
            f"yaw={best_generation['aggregate']['mean_abs_yaw_error_degrees']:.2f} "
            f"failure={best_generation['aggregate']['catastrophic_failure_rate']:.3f}",
            flush=True,
        )

        elites = scored_population[: max(1, min(args.elite_count, len(scored_population)))]
        next_population = [SearchCandidate(**elite["candidate"]) for elite in elites]
        while len(next_population) < args.population_size:
            parent_a = tournament_select(scored_population, rng)
            parent_b = tournament_select(scored_population, rng)
            child = crossover(
                parent_a,
                parent_b,
                rng=rng,
                metric_name=metric_name,
                child_seed=next_seed,
                settings=settings,
            )
            next_seed += 1
            child = mutate(
                child,
                rng=rng,
                metric_name=metric_name,
                mutation_rate=args.mutation_rate,
                child_seed=next_seed,
                settings=settings,
            )
            next_seed += 1
            next_population.append(child)
        population = next_population[: args.population_size]

    payload = {
        "config": str(args.config),
        "metric_name": metric_name,
        "manifests": [str(path) for path in args.manifest],
        "population_size": args.population_size,
        "generations": args.generations,
        "elite_count": args.elite_count,
        "mutation_rate": args.mutation_rate,
        "history": history,
        "best_overall": best_overall,
        "objective_plot_png": str(plotter.objective_png_path),
        "objective_plot_pdf": str(plotter.objective_pdf_path),
        "error_plot_png": str(plotter.error_png_path),
        "error_plot_pdf": str(plotter.error_pdf_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    plotter.save()
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {plotter.objective_png_path}", flush=True)
    print(f"wrote {plotter.objective_pdf_path}", flush=True)
    print(f"wrote {plotter.error_png_path}", flush=True)
    print(f"wrote {plotter.error_pdf_path}", flush=True)


if __name__ == "__main__":
    main()
