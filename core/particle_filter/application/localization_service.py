"""Coordinates the live localization loop across ROS input, rendering, and UI output."""

from __future__ import annotations

import random

from core.config import MeasurementSettings, MotionNoiseSettings, TurtleBotLocalizationConfig
from core.particle_filter.application.command_handler import LocalizationCommandHandler
from core.particle_filter.application.diagnostics import LocalizationDiagnosticsFormatter
from core.particle_filter.application.runtime_state import LocalizationRuntimeState
from core.particle_filter.application.snapshot_builder import LocalizationSnapshotBuilder
from core.particle_filter.application.step_engine import LocalizationStepEngine
from core.particle_filter.domain.motion_model import TurtleBotMotionModel
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.particle_filter.infrastructure.ros.turtlebot_observation_source import TurtleBotObservationSource
from core.particle_filter.infrastructure.visualization.control_client import ControlCommandClient
from core.particle_filter.infrastructure.visualization.publisher import VisualizationPublisher


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

        particle_filter_config = TurtleBotParticleFilterConfig(
            particle_count=settings.particle_filter.particle_count,
            resample_threshold_ratio=settings.particle_filter.resample_threshold_ratio,
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
        particle_filter.initialize(prior)
        self._runtime_state = LocalizationRuntimeState(
            particle_filter=particle_filter,
            particle_filter_config=particle_filter_config,
            prior=prior,
            motion_noise=motion_noise,
            measurement=measurement,
            motion_model=motion_model,
            rng=rng,
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
            f"particles={len(self._runtime_state.particle_filter.particles)} | "
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

        self._runtime_state.last_processed_image_stamp = latest_image_stamp
        previous_odometry_pose = self._runtime_state.previous_odometry_pose
        if command_effect.reset_applied:
            previous_odometry_pose = observation.odometry_pose

        step_result = self._step_engine.run_step(
            particle_filter=self._runtime_state.particle_filter,
            observation=observation,
            previous_odometry_pose=previous_odometry_pose,
        )
        self._runtime_state.previous_odometry_pose = step_result.previous_odometry_pose

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
        print(status_line)
        self._runtime_state.update_count += 1

        if self._runtime_state.step_once_requested:
            self._runtime_state.step_once_requested = False
