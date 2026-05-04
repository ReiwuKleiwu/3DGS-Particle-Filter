from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config_loader import MeasurementSettings, MotionNoiseSettings, TurtleBotLocalizationConfig
from particle_filter.domain.motion_model import TurtleBotMotionModel
from particle_filter.domain.odometry import compute_odometry_delta_in_robot_frame
from particle_filter.domain.particle_filter import TurtleBotParticleFilter, TurtleBotParticleFilterConfig
from particle_filter.domain.pose import Pose2D, Pose2DPrior
from particle_filter.infrastructure.renderer.splat_renderer_client import SplatRendererClient
from particle_filter.infrastructure.ros.observation import TurtleBotObservation
from particle_filter.infrastructure.ros.turtlebot_observation_source import TurtleBotObservationSource
from particle_filter.infrastructure.visualization.control_client import ControlCommandClient
from particle_filter.infrastructure.visualization.models import (
    VisualizationFilterState,
    VisualizationParticle,
    VisualizationSnapshot,
)
from particle_filter.infrastructure.visualization.publisher import VisualizationPublisher


@dataclass(frozen=True)
class CommandEffect:
    reset_applied: bool = False
    reprocess_current_observation: bool = False


class TurtleBotLocalizationService:
    def __init__(
        self,
        *,
        settings: TurtleBotLocalizationConfig,
        observation_source: TurtleBotObservationSource,
        renderer_client: SplatRendererClient,
        visualization_publisher: VisualizationPublisher,
        control_command_client: ControlCommandClient,
    ) -> None:
        self._settings = settings
        self._observation_source = observation_source
        self._renderer_client = renderer_client
        self._visualization_publisher = visualization_publisher
        self._control_command_client = control_command_client

        self._current_particle_filter_config = TurtleBotParticleFilterConfig(
            particle_count=settings.particle_filter.particle_count,
            resample_threshold_ratio=settings.particle_filter.resample_threshold_ratio,
        )
        self._current_prior = settings.initial_pose_prior
        self._current_motion_noise = MotionNoiseSettings(
            x_meters=settings.motion_noise.x_meters,
            y_meters=settings.motion_noise.y_meters,
            yaw_radians=settings.motion_noise.yaw_radians,
        )
        self._current_measurement = MeasurementSettings(
            metric_name=settings.measurement.metric_name,
            temperature=settings.measurement.temperature,
            packed=settings.measurement.packed,
            radius_clip=settings.measurement.radius_clip,
            hybrid_ssim_weight=settings.measurement.hybrid_ssim_weight,
            hybrid_l1_weight=settings.measurement.hybrid_l1_weight,
            hybrid_gradient_weight=settings.measurement.hybrid_gradient_weight,
        )
        self._paused = False
        self._step_once_requested = False

        self._motion_model = TurtleBotMotionModel(
            noise_x=self._current_motion_noise.x_meters,
            noise_y=self._current_motion_noise.y_meters,
            noise_yaw=self._current_motion_noise.yaw_radians,
        )
        self._particle_filter = TurtleBotParticleFilter(
            config=self._current_particle_filter_config,
            motion_model=self._motion_model,
        )
        self._particle_filter.initialize(self._current_prior)

        self._previous_odometry_pose = None
        self._last_processed_image_stamp = None
        self._update_count = 0
        self._best_render_path = Path(__file__).resolve().parents[2] / "renders" / "render.png"
        self._best_render_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        renderer_status = self._renderer_client.wait_until_ready()
        print(
            f"Renderer ready at {self._settings.renderer.base_url} | "
            f"splat={renderer_status.get('splat_path')} | gaussians={renderer_status.get('gaussians')}"
        )

        self._observation_source.wait_until_ready(self._settings.runtime.observation_ready_timeout_seconds)
        print(
            "Particle filter initialized | "
            f"particles={len(self._particle_filter.particles)} | "
            f"prior x={self._current_prior.mean.x:.3f}, "
            f"y={self._current_prior.mean.y:.3f}, "
            f"yaw={self._current_prior.mean.yaw:.3f}"
        )

        while self._observation_source.is_running():
            self._process_next_observation()

    def shutdown(self) -> None:
        self._visualization_publisher.close()

    def _process_next_observation(self) -> None:
        self._observation_source.spin_once(self._settings.runtime.spin_timeout_seconds)

        latest_image_stamp = self._observation_source.latest_image_stamp()
        if latest_image_stamp is None:
            return

        observation = self._observation_source.read_latest_observation()
        command_effect = self._apply_pending_control_command(observation)

        has_new_image = latest_image_stamp != self._last_processed_image_stamp
        if self._paused and not self._step_once_requested:
            return
        if not has_new_image and not command_effect.reprocess_current_observation:
            return

        self._last_processed_image_stamp = latest_image_stamp

        if not command_effect.reset_applied:
            self._predict_from_odometry(observation)

        score_result = self._renderer_client.score_particles(
            particle_poses=[particle.pose for particle in self._particle_filter.particles],
            observation=observation,
            metric_name=self._current_measurement.metric_name,
            packed=self._current_measurement.packed,
            radius_clip=self._current_measurement.radius_clip,
            hybrid_ssim_weight=self._current_measurement.hybrid_ssim_weight,
            hybrid_l1_weight=self._current_measurement.hybrid_l1_weight,
            hybrid_gradient_weight=self._current_measurement.hybrid_gradient_weight,
        )
        self._best_render_path.write_bytes(score_result.best_render_png_bytes)

        self._particle_filter.update_from_measurement_errors(
            score_result.errors,
            temperature=self._current_measurement.temperature,
        )
        effective_particle_count = self._particle_filter.effective_particle_count()
        resampled = self._particle_filter.resample_if_needed()
        estimated_pose = self._particle_filter.estimate_pose()
        best_score = score_result.errors[score_result.best_index]

        self._publish_visualization_snapshot(
            observation=observation,
            estimated_pose=estimated_pose,
            effective_particle_count=effective_particle_count,
            best_particle_index=score_result.best_index,
            best_score=best_score,
            render_and_score_milliseconds=score_result.elapsed_milliseconds,
            resampled=resampled,
            best_render_png_bytes=score_result.best_render_png_bytes,
        )

        status_line = (
            f"update {self._update_count:04d} | "
            f"estimate x={estimated_pose.x:.3f}, y={estimated_pose.y:.3f}, yaw={estimated_pose.yaw:.3f} | "
            f"best_index={score_result.best_index} | "
            f"render+score={score_result.elapsed_milliseconds:.1f} ms | "
            f"resampled={'yes' if resampled else 'no'} | "
            f"paused={'yes' if self._paused else 'no'}"
        )

        if observation.map_pose is not None:
            pose_error = observation.pose_error_against(estimated_pose)
            status_line += (
                f" | eval dx={pose_error.x:.3f}, "
                f"dy={pose_error.y:.3f}, "
                f"dyaw={pose_error.yaw:.3f}"
            )

        print(status_line)
        self._update_count += 1

        if self._step_once_requested:
            self._step_once_requested = False

    def _apply_pending_control_command(self, observation: TurtleBotObservation) -> CommandEffect:
        command = self._control_command_client.poll_next_command()
        if command is None:
            return CommandEffect()

        command_type = command.get("type")
        if command_type == "reset_particle_filter":
            return self._apply_reset_command(command, observation)
        if command_type == "set_particle_filter_parameters":
            return self._apply_parameter_update_command(command)
        if command_type == "pause_particle_filter":
            self._paused = True
            self._step_once_requested = False
            print("Particle filter paused from frontend")
            return CommandEffect()
        if command_type == "resume_particle_filter":
            self._paused = False
            self._step_once_requested = False
            print("Particle filter resumed from frontend")
            return CommandEffect()
        if command_type == "step_particle_filter":
            self._paused = True
            self._step_once_requested = True
            print("Particle filter step requested from frontend")
            return CommandEffect()

        print(f"Ignoring unknown control command: {command_type}")
        return CommandEffect()

    def _apply_reset_command(self, command: dict, observation: TurtleBotObservation) -> CommandEffect:
        prior_payload = command["prior"]
        mean_payload = prior_payload["mean"]
        prior = Pose2DPrior(
            mean=Pose2D(
                x=float(mean_payload["x"]),
                y=float(mean_payload["y"]),
                yaw=float(mean_payload["yaw"]),
            ),
            sigma_x=float(prior_payload["sigma_x"]),
            sigma_y=float(prior_payload["sigma_y"]),
            sigma_yaw=float(prior_payload["sigma_yaw"]),
        )
        self._current_prior = prior
        self._particle_filter.initialize(prior)
        self._previous_odometry_pose = observation.odometry_pose
        print(
            "Particle filter reset from frontend | "
            f"prior x={prior.mean.x:.3f}, y={prior.mean.y:.3f}, yaw={prior.mean.yaw:.3f} | "
            f"sigma_x={prior.sigma_x:.3f}, sigma_y={prior.sigma_y:.3f}, sigma_yaw={prior.sigma_yaw:.3f}"
        )
        return CommandEffect(reset_applied=True, reprocess_current_observation=True)

    def _apply_parameter_update_command(self, command: dict) -> CommandEffect:
        reprocess_current_observation = False
        changed_fields: list[str] = []

        particle_count = command.get("particle_count")
        resample_threshold_ratio = command.get("resample_threshold_ratio")
        if particle_count is not None or resample_threshold_ratio is not None:
            self._particle_filter.reconfigure(
                particle_count=int(particle_count) if particle_count is not None else None,
                resample_threshold_ratio=float(resample_threshold_ratio) if resample_threshold_ratio is not None else None,
            )
            self._current_particle_filter_config = self._particle_filter.config
            changed_fields.append(
                f"particles={self._current_particle_filter_config.particle_count}, "
                f"resample={self._current_particle_filter_config.resample_threshold_ratio:.2f}"
            )
            if particle_count is not None:
                reprocess_current_observation = True

        temperature = command.get("temperature")
        if temperature is not None:
            self._current_measurement = MeasurementSettings(
                metric_name=self._current_measurement.metric_name,
                temperature=float(temperature),
                packed=self._current_measurement.packed,
                radius_clip=self._current_measurement.radius_clip,
                hybrid_ssim_weight=self._current_measurement.hybrid_ssim_weight,
                hybrid_l1_weight=self._current_measurement.hybrid_l1_weight,
                hybrid_gradient_weight=self._current_measurement.hybrid_gradient_weight,
            )
            changed_fields.append(f"temperature={self._current_measurement.temperature:.3f}")
            reprocess_current_observation = True

        motion_noise = command.get("motion_noise")
        if motion_noise is not None:
            self._current_motion_noise = MotionNoiseSettings(
                x_meters=float(motion_noise.get("x_meters", self._current_motion_noise.x_meters)),
                y_meters=float(motion_noise.get("y_meters", self._current_motion_noise.y_meters)),
                yaw_radians=float(motion_noise.get("yaw_radians", self._current_motion_noise.yaw_radians)),
            )
            self._motion_model.set_noise(
                noise_x=self._current_motion_noise.x_meters,
                noise_y=self._current_motion_noise.y_meters,
                noise_yaw=self._current_motion_noise.yaw_radians,
            )
            changed_fields.append(
                "motion_noise=("
                f"{self._current_motion_noise.x_meters:.3f}, "
                f"{self._current_motion_noise.y_meters:.3f}, "
                f"{self._current_motion_noise.yaw_radians:.3f})"
            )

        if changed_fields:
            print("Particle filter parameters updated from frontend | " + " | ".join(changed_fields))

        return CommandEffect(reprocess_current_observation=reprocess_current_observation)

    def _predict_from_odometry(self, observation: TurtleBotObservation) -> None:
        current_odometry_pose = observation.odometry_pose
        if current_odometry_pose is None:
            return

        if self._previous_odometry_pose is not None:
            odometry_delta = compute_odometry_delta_in_robot_frame(
                self._previous_odometry_pose,
                current_odometry_pose,
            )
            self._particle_filter.predict_from_odometry(odometry_delta)

        self._previous_odometry_pose = current_odometry_pose

    def _publish_visualization_snapshot(
        self,
        *,
        observation: TurtleBotObservation,
        estimated_pose,
        effective_particle_count: float,
        best_particle_index: int,
        best_score: float,
        render_and_score_milliseconds: float,
        resampled: bool,
        best_render_png_bytes: bytes,
    ) -> None:
        snapshot = VisualizationSnapshot(
            update_index=self._update_count,
            image_stamp_seconds=observation.image_stamp_seconds,
            image_stamp_nanoseconds=observation.image_stamp_nanoseconds,
            particles=[
                VisualizationParticle(
                    x=particle.pose.x,
                    y=particle.pose.y,
                    yaw=particle.pose.yaw,
                    weight=particle.weight,
                )
                for particle in self._particle_filter.particles
            ],
            estimated_pose=estimated_pose,
            ground_truth_pose=observation.map_pose,
            best_particle_index=best_particle_index,
            best_score=best_score,
            effective_particle_count=effective_particle_count,
            render_and_score_milliseconds=render_and_score_milliseconds,
            resampled=resampled,
            observation_image_rgb=observation.image_rgb,
            best_render_png_bytes=best_render_png_bytes,
            filter_state=VisualizationFilterState(
                particle_count=self._current_particle_filter_config.particle_count,
                resample_threshold_ratio=self._current_particle_filter_config.resample_threshold_ratio,
                temperature=self._current_measurement.temperature,
                motion_noise_x_meters=self._current_motion_noise.x_meters,
                motion_noise_y_meters=self._current_motion_noise.y_meters,
                motion_noise_yaw_radians=self._current_motion_noise.yaw_radians,
                paused=self._paused,
            ),
        )
        self._visualization_publisher.publish(snapshot)
