"""Coordinates the live localization loop across ROS input, rendering, and UI output."""

from __future__ import annotations

import math
import random
from pathlib import Path

from core.config import MeasurementSettings, MotionNoiseSettings, TurtleBotLocalizationConfig
from core.particle_filter.application.adaptive_particle_count import AdaptiveParticleCountController
from core.particle_filter.application.command_handler import LocalizationCommandHandler
from core.particle_filter.application.diagnostics import LocalizationDiagnosticsFormatter
from core.particle_filter.application.runtime_state import LocalizationRuntimeState
from core.particle_filter.application.snapshot_builder import LocalizationSnapshotBuilder
from core.particle_filter.application.step_engine import LocalizationStepEngine
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.odometry import compute_odometry_delta_in_robot_frame
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from core.particle_filter.domain.recovery import AugmentedMclRecoveryTracker
from core.particle_filter.infrastructure.map import FreeSpacePoseSampler
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.particle_filter.infrastructure.ros.turtlebot_observation_source import TurtleBotObservationSource
from core.particle_filter.infrastructure.visualization.control_client import ControlCommandClient
from core.particle_filter.infrastructure.visualization.publisher import VisualizationPublisher


DEFAULT_MAP_YAML_PATH = Path(__file__).resolve().parents[3] / "map.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TurtleBotLocalizationService:
    def __init__(
        self,
        *,
        settings: TurtleBotLocalizationConfig,
        observation_source: TurtleBotObservationSource,
        renderer_client: RendererServiceClient,
        visualization_publisher: VisualizationPublisher,
        control_command_client: ControlCommandClient,
    ) -> None:
        self._settings = settings
        self._observation_source = observation_source
        self._renderer_client = renderer_client
        self._visualization_publisher = visualization_publisher
        self._control_command_client = control_command_client

        adaptive_particle_count = settings.adaptive_particle_count
        initial_particle_count = (
            adaptive_particle_count.max_particle_count
            if adaptive_particle_count.enabled and adaptive_particle_count.max_particle_count is not None
            else settings.particle_filter.particle_count
        )
        particle_filter_config = TurtleBotParticleFilterConfig(
            particle_count=initial_particle_count,
            resample_threshold_ratio=settings.particle_filter.resample_threshold_ratio,
            roughening_enabled=settings.particle_filter.roughening_enabled,
            roughening_mode=settings.particle_filter.roughening_mode,
            roughening_ratio=settings.particle_filter.roughening_ratio,
            roughening_sigma_x=settings.particle_filter.roughening_sigma_x,
            roughening_sigma_y=settings.particle_filter.roughening_sigma_y,
            roughening_sigma_yaw=settings.particle_filter.roughening_sigma_yaw,
        )
        prior = settings.initial_pose_prior
        motion_noise = MotionNoiseSettings(
            x_meters=settings.motion_noise.x_meters,
            y_meters=settings.motion_noise.y_meters,
            yaw_radians=settings.motion_noise.yaw_radians,
        )
        measurement = MeasurementSettings(
            metric_name=settings.measurement.metric_name,
            temperature=settings.measurement.temperature,
            packed=settings.measurement.packed,
            radius_clip=settings.measurement.radius_clip,
            hybrid_ssim_weight=settings.measurement.hybrid_ssim_weight,
            hybrid_l1_weight=settings.measurement.hybrid_l1_weight,
            hybrid_gradient_weight=settings.measurement.hybrid_gradient_weight,
            lpips_top_k=settings.measurement.lpips_top_k,
            lpips_weight=settings.measurement.lpips_weight,
            lpips_net=settings.measurement.lpips_net,
        )
        rng = random.Random(settings.runtime.random_seed)

        motion_model = TurtleBotMotionModel(
            noise_x=motion_noise.x_meters,
            noise_y=motion_noise.y_meters,
            noise_yaw=motion_noise.yaw_radians,
            rng=rng,
        )
        particle_filter = TurtleBotParticleFilter(
            config=particle_filter_config,
            motion_model=motion_model,
            rng=rng,
        )

        map_yaml_path = Path(settings.map.yaml_path)
        if not map_yaml_path.is_absolute():
            map_yaml_path = PROJECT_ROOT / map_yaml_path
        free_space_sampler = FreeSpacePoseSampler.from_map_yaml(
            map_yaml_path,
            global_yaw_uniform=settings.initialization.global_yaw_uniform,
        )
        global_pose_sampler = lambda: free_space_sampler.sample_pose(rng=rng)

        if settings.initialization.mode == "global":
            particle_filter.initialize_global(global_pose_sampler)
        else:
            particle_filter.initialize(prior)

        recovery_tracker = AugmentedMclRecoveryTracker(settings.recovery)
        adaptive_particle_count_controller = AdaptiveParticleCountController(
            adaptive_particle_count,
            configured_particle_count=particle_filter_config.particle_count,
        )
        self._runtime_state = LocalizationRuntimeState(
            particle_filter=particle_filter,
            particle_filter_config=particle_filter_config,
            prior=prior,
            motion_noise=motion_noise,
            measurement=measurement,
            motion_model=motion_model,
            rng=rng,
            localization_mode=settings.initialization.mode,
            recovery_tracker=recovery_tracker,
            adaptive_particle_count_settings=adaptive_particle_count,
            adaptive_particle_count_controller=adaptive_particle_count_controller,
            global_pose_sampler=global_pose_sampler,
        )
        self._step_engine = LocalizationStepEngine(renderer_client, measurement)
        self._command_handler = LocalizationCommandHandler(self._runtime_state)
        self._snapshot_builder = LocalizationSnapshotBuilder()
        self._diagnostics_formatter = LocalizationDiagnosticsFormatter()

    def run(self) -> None:
        """Starts the live localization loop after the renderer and ROS inputs are ready."""
        renderer_status = self._renderer_client.wait_until_ready()
        actual_backend = renderer_status.get("backend")
        configured_backend = self._settings.renderer.backend
        if actual_backend is not None and actual_backend != configured_backend:
            print(
                f"Warning: renderer config backend is '{configured_backend}' but service reports '{actual_backend}'."
            )
        print(
            f"Renderer ready at {self._settings.renderer.base_url} | "
            f"backend={actual_backend} | "
            f"splat={renderer_status.get('splat_path')} | gaussians={renderer_status.get('gaussians')}"
        )

        self._observation_source.wait_until_ready(self._settings.runtime.observation_ready_timeout_seconds)
        print(
            "Particle filter initialized | "
            f"mode={self._runtime_state.localization_mode} | "
            f"particles={len(self._runtime_state.particle_filter.particles)} | "
            f"roughening={'on' if self._runtime_state.particle_filter_config.roughening_enabled else 'off'} | "
            f"roughening_mode={self._runtime_state.particle_filter_config.roughening_mode} | "
            f"roughening_ratio={self._runtime_state.particle_filter_config.roughening_ratio:.3f} | "
            f"prior x={self._runtime_state.prior.mean.x:.3f}, "
            f"y={self._runtime_state.prior.mean.y:.3f}, "
            f"yaw={self._runtime_state.prior.mean.yaw:.3f}"
        )

        while self._observation_source.is_running():
            self._process_next_observation()

    def shutdown(self) -> None:
        """Releases external resources owned by the live localization service."""
        self._visualization_publisher.close()

    def _process_next_observation(self) -> None:
        """Consumes the latest observation, applies commands, and performs one live filter update."""
        self._observation_source.spin_once(self._settings.runtime.spin_timeout_seconds)

        latest_image_stamp = self._observation_source.latest_image_stamp()
        if latest_image_stamp is None:
            return

        observation = self._observation_source.read_latest_observation()
        command_effect = self._command_handler.apply(
            self._control_command_client.poll_next_command(),
            observation,
        )
        self._step_engine.set_measurement(self._runtime_state.measurement)

        has_new_image = latest_image_stamp != self._runtime_state.last_processed_image_stamp
        if self._runtime_state.paused and not self._runtime_state.step_once_requested:
            return
        if not has_new_image and not command_effect.reprocess_current_observation:
            return

        if (
            has_new_image
            and not command_effect.reprocess_current_observation
            and not self._runtime_state.step_once_requested
            and self._should_suspend_for_stationary_observation(observation)
        ):
            return

        self._runtime_state.last_processed_image_stamp = latest_image_stamp
        previous_odometry_pose = self._runtime_state.previous_odometry_pose
        if command_effect.reset_applied:
            previous_odometry_pose = observation.odometry_pose

        step_result = self._step_engine.run_step(
            particle_filter=self._runtime_state.particle_filter,
            observation=observation,
            previous_odometry_pose=previous_odometry_pose,
            recovery_tracker=self._runtime_state.recovery_tracker,
            random_pose_sampler=self._runtime_state.global_pose_sampler,
        )
        self._runtime_state.previous_odometry_pose = step_result.previous_odometry_pose
        self._apply_adaptive_particle_count(step_result)

        snapshot = self._snapshot_builder.build(
            runtime_state=self._runtime_state,
            observation=observation,
            step_result=step_result,
        )
        self._visualization_publisher.publish(snapshot)

        status_line = self._diagnostics_formatter.format(
            update_count=self._runtime_state.update_count,
            paused=self._runtime_state.paused,
            observation=observation,
            step_result=step_result,
        )
        print(
            status_line
            + f" | mode={self._runtime_state.localization_mode}"
            + f" | rand={step_result.random_particle_ratio:.3f}"
            + f" | injected={step_result.random_particle_count}"
            + f" | roughened={step_result.roughening_particle_count}"
            + self._adaptive_status_suffix()
        )
        self._runtime_state.update_count += 1

        if self._runtime_state.step_once_requested:
            self._runtime_state.step_once_requested = False

    def _should_suspend_for_stationary_observation(self, observation) -> bool:
        if not self._settings.runtime.suspend_updates_when_stationary:
            return False

        previous_odometry_pose = self._runtime_state.previous_odometry_pose
        current_odometry_pose = observation.odometry_pose
        if previous_odometry_pose is None or current_odometry_pose is None:
            return False

        odometry_delta = compute_odometry_delta_in_robot_frame(previous_odometry_pose, current_odometry_pose)
        translation_magnitude = math.hypot(odometry_delta.forward_meters, odometry_delta.lateral_meters)
        return (
            translation_magnitude < self._settings.runtime.stationary_translation_threshold_meters
            and abs(odometry_delta.yaw_radians) < self._settings.runtime.stationary_yaw_threshold_radians
        )

    def _apply_adaptive_particle_count(self, step_result) -> None:
        controller = self._runtime_state.adaptive_particle_count_controller
        decision = controller.update(
            particles=self._runtime_state.particle_filter.particles,
            current_particle_count=len(self._runtime_state.particle_filter.particles),
            best_score=step_result.best_score,
            median_score=step_result.median_score,
            random_particle_count=step_result.random_particle_count,
        )
        if (
            self._runtime_state.adaptive_particle_count_settings.enabled
            and decision.resize_reason is not None
            and decision.target_particle_count != len(self._runtime_state.particle_filter.particles)
        ):
            self._runtime_state.particle_filter.reconfigure(particle_count=decision.target_particle_count)
            self._runtime_state.particle_filter_config = self._runtime_state.particle_filter.config

    def _adaptive_status_suffix(self) -> str:
        status = self._runtime_state.adaptive_particle_count_controller.status()
        if not status.enabled:
            return ""
        return (
            f" | adaptive_particles={len(self._runtime_state.particle_filter.particles)}/{status.max_particle_count}"
            f" | spread={status.xy_spread_meters:.3f}"
            f" | yaw_spread={status.yaw_spread_radians:.3f}"
        )
