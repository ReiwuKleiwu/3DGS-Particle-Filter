"""Applies frontend control commands to the live localization runtime state."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.models import MeasurementSettings, MotionNoiseSettings
from core.particle_filter.application.runtime_state import LocalizationRuntimeState
from core.particle_filter.domain.pose import Pose2D, Pose2DPrior
from core.particle_filter.infrastructure.ros.observation import TurtleBotObservation


@dataclass(frozen=True)
class CommandEffect:
    reset_applied: bool = False
    reprocess_current_observation: bool = False


class LocalizationCommandHandler:
    """Mutates live localization runtime state in response to frontend commands."""

    def __init__(self, runtime_state: LocalizationRuntimeState) -> None:
        self._runtime_state = runtime_state

    def apply(self, command: dict | None, observation: TurtleBotObservation) -> CommandEffect:
        """Applies one frontend command to runtime state and reports how the loop should react."""
        if command is None:
            return CommandEffect()

        command_type = command.get("type")
        if command_type == "reset_particle_filter":
            return self._apply_reset_command(command, observation)
        if command_type == "global_reset_particle_filter":
            return self._apply_global_reset_command(observation)
        if command_type == "set_localization_mode":
            return self._apply_localization_mode_command(command)
        if command_type == "set_particle_filter_parameters":
            return self._apply_parameter_update_command(command)
        if command_type == "pause_particle_filter":
            self._runtime_state.paused = True
            self._runtime_state.step_once_requested = False
            print("Particle filter paused from frontend")
            return CommandEffect()
        if command_type == "resume_particle_filter":
            self._runtime_state.paused = False
            self._runtime_state.step_once_requested = False
            print("Particle filter resumed from frontend")
            return CommandEffect()
        if command_type == "step_particle_filter":
            self._runtime_state.paused = True
            self._runtime_state.step_once_requested = True
            print("Particle filter step requested from frontend")
            return CommandEffect()

        print(f"Ignoring unknown control command: {command_type}")
        return CommandEffect()

    def _apply_reset_command(self, command: dict, observation: TurtleBotObservation) -> CommandEffect:
        """Reinitializes the particle filter from a user-specified prior."""
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
        self._runtime_state.localization_mode = "local"
        self._runtime_state.prior = prior
        self._runtime_state.particle_filter.initialize(prior)
        self._runtime_state.recovery_tracker.reset()
        self._runtime_state.previous_odometry_pose = observation.odometry_pose
        print(
            "Particle filter reset from frontend | "
            f"mode=local | prior x={prior.mean.x:.3f}, y={prior.mean.y:.3f}, yaw={prior.mean.yaw:.3f} | "
            f"sigma_x={prior.sigma_x:.3f}, sigma_y={prior.sigma_y:.3f}, sigma_yaw={prior.sigma_yaw:.3f}"
        )
        return CommandEffect(reset_applied=True, reprocess_current_observation=True)

    def _apply_global_reset_command(self, observation: TurtleBotObservation) -> CommandEffect:
        if self._runtime_state.global_pose_sampler is None:
            print("Ignoring global reset because no global pose sampler is configured")
            return CommandEffect()

        self._runtime_state.localization_mode = "global"
        self._runtime_state.particle_filter.initialize_global(self._runtime_state.global_pose_sampler)
        self._runtime_state.recovery_tracker.reset()
        self._runtime_state.previous_odometry_pose = observation.odometry_pose
        print("Particle filter reset from frontend | mode=global | samples drawn from map free space")
        return CommandEffect(reset_applied=True, reprocess_current_observation=True)

    def _apply_localization_mode_command(self, command: dict) -> CommandEffect:
        mode = str(command.get("mode", "")).strip().lower()
        if mode not in {"local", "global"}:
            print(f"Ignoring invalid localization mode command: {mode!r}")
            return CommandEffect()
        self._runtime_state.localization_mode = mode
        print(f"Particle filter localization mode set from frontend | mode={mode}")
        return CommandEffect()

    def _apply_parameter_update_command(self, command: dict) -> CommandEffect:
        """Applies mutable runtime parameter changes without rebuilding the whole service."""
        reprocess_current_observation = False
        changed_fields: list[str] = []

        particle_count = command.get("particle_count")
        resample_threshold_ratio = command.get("resample_threshold_ratio")
        if particle_count is not None or resample_threshold_ratio is not None:
            self._runtime_state.particle_filter.reconfigure(
                particle_count=int(particle_count) if particle_count is not None else None,
                resample_threshold_ratio=float(resample_threshold_ratio) if resample_threshold_ratio is not None else None,
            )
            self._runtime_state.particle_filter_config = self._runtime_state.particle_filter.config
            changed_fields.append(
                f"particles={self._runtime_state.particle_filter_config.particle_count}, "
                f"resample={self._runtime_state.particle_filter_config.resample_threshold_ratio:.2f}"
            )
            if particle_count is not None:
                reprocess_current_observation = True

        temperature = command.get("temperature")
        if temperature is not None:
            current = self._runtime_state.measurement
            self._runtime_state.measurement = MeasurementSettings(
                metric_name=current.metric_name,
                temperature=float(temperature),
                packed=current.packed,
                radius_clip=current.radius_clip,
                hybrid_ssim_weight=current.hybrid_ssim_weight,
                hybrid_l1_weight=current.hybrid_l1_weight,
                hybrid_gradient_weight=current.hybrid_gradient_weight,
                lpips_top_k=current.lpips_top_k,
                lpips_weight=current.lpips_weight,
                lpips_net=current.lpips_net,
            )
            changed_fields.append(f"temperature={self._runtime_state.measurement.temperature:.3f}")
            reprocess_current_observation = True

        motion_noise = command.get("motion_noise")
        if motion_noise is not None:
            current = self._runtime_state.motion_noise
            self._runtime_state.motion_noise = MotionNoiseSettings(
                x_meters=float(motion_noise.get("x_meters", current.x_meters)),
                y_meters=float(motion_noise.get("y_meters", current.y_meters)),
                yaw_radians=float(motion_noise.get("yaw_radians", current.yaw_radians)),
            )
            self._runtime_state.motion_model.set_noise(
                noise_x=self._runtime_state.motion_noise.x_meters,
                noise_y=self._runtime_state.motion_noise.y_meters,
                noise_yaw=self._runtime_state.motion_noise.yaw_radians,
            )
            changed_fields.append(
                "motion_noise=("
                f"{self._runtime_state.motion_noise.x_meters:.3f}, "
                f"{self._runtime_state.motion_noise.y_meters:.3f}, "
                f"{self._runtime_state.motion_noise.yaw_radians:.3f})"
            )

        if changed_fields:
            print("Particle filter parameters updated from frontend | " + " | ".join(changed_fields))

        return CommandEffect(reprocess_current_observation=reprocess_current_observation)
