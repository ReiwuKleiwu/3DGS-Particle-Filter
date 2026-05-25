"""Section-specific configuration parsers used by the top-level config loader."""

from __future__ import annotations

from typing import Any

from core.config.models import (
    InitializationSettings,
    MeasurementSettings,
    MotionNoiseSettings,
    RecoverySettings,
    RuntimeSettings,
    TurtleBotLocalizationConfig,
)
from core.particle_filter.domain.particle_filter import TurtleBotParticleFilterConfig
from core.particle_filter.domain.pose import Pose2D, Pose2DPrior
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceSettings
from core.particle_filter.infrastructure.ros.observation import RosTopicSettings
from core.particle_filter.infrastructure.visualization.control_client import ControlCommandClientSettings
from core.particle_filter.infrastructure.visualization.publisher import VisualizationPublisherSettings


def load_renderer_settings(raw: dict[str, Any]) -> RendererServiceSettings:
    """Parses renderer-service settings from one config section."""
    defaults = RendererServiceSettings()
    return RendererServiceSettings(
        backend=raw.get("backend", defaults.backend),
        base_url=raw.get("base_url", defaults.base_url),
        wait_timeout_seconds=float(raw.get("wait_timeout_seconds", defaults.wait_timeout_seconds)),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
        poll_interval_seconds=float(raw.get("poll_interval_seconds", defaults.poll_interval_seconds)),
        score_batch_size=int(raw.get("score_batch_size", defaults.score_batch_size)),
        include_best_render_preview=bool(
            raw.get("include_best_render_preview", defaults.include_best_render_preview)
        ),
    )


def load_ros_topic_settings(raw: dict[str, Any]) -> RosTopicSettings:
    """Parses ROS topic and TF settings for the live observation source."""
    defaults = RosTopicSettings()
    return RosTopicSettings(
        image_topic=raw.get("image_topic", defaults.image_topic),
        camera_info_topic=raw.get("camera_info_topic", defaults.camera_info_topic),
        odometry_topic=raw.get("odometry_topic", defaults.odometry_topic),
        amcl_pose_topic=raw.get("amcl_pose_topic", defaults.amcl_pose_topic),
        map_frame=raw.get("map_frame", defaults.map_frame),
        base_frame=raw.get("base_frame", defaults.base_frame),
        tf_lookup_mode=raw.get("tf_lookup_mode", defaults.tf_lookup_mode),
        tf_timeout_seconds=float(raw.get("tf_timeout_seconds", defaults.tf_timeout_seconds)),
        require_odometry=bool(raw.get("require_odometry", defaults.require_odometry)),
    )


def load_runtime_settings(raw: dict[str, Any]) -> RuntimeSettings:
    """Parses general runtime loop settings such as timeouts and random seed."""
    defaults = RuntimeSettings()
    return RuntimeSettings(
        observation_ready_timeout_seconds=float(
            raw.get("observation_ready_timeout_seconds", defaults.observation_ready_timeout_seconds)
        ),
        spin_timeout_seconds=float(raw.get("spin_timeout_seconds", defaults.spin_timeout_seconds)),
        random_seed=None if raw.get("random_seed", defaults.random_seed) is None else int(raw.get("random_seed")),
        suspend_updates_when_stationary=bool(
            raw.get("suspend_updates_when_stationary", defaults.suspend_updates_when_stationary)
        ),
        stationary_translation_threshold_meters=float(
            raw.get(
                "stationary_translation_threshold_meters",
                defaults.stationary_translation_threshold_meters,
            )
        ),
        stationary_yaw_threshold_radians=float(
            raw.get(
                "stationary_yaw_threshold_radians",
                defaults.stationary_yaw_threshold_radians,
            )
        ),
    )


def load_visualization_settings(raw: dict[str, Any]) -> VisualizationPublisherSettings:
    """Parses visualization publishing settings for the frontend bridge."""
    defaults = VisualizationPublisherSettings()
    return VisualizationPublisherSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        publish_url=raw.get("publish_url", defaults.publish_url),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
        observation_jpeg_quality=int(raw.get("observation_jpeg_quality", defaults.observation_jpeg_quality)),
    )


def load_control_settings(raw: dict[str, Any]) -> ControlCommandClientSettings:
    """Parses frontend control-polling settings for live runtime commands."""
    defaults = ControlCommandClientSettings()
    return ControlCommandClientSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        poll_url=raw.get("poll_url", defaults.poll_url),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
    )


def load_particle_filter_settings(raw: dict[str, Any]) -> TurtleBotParticleFilterConfig:
    """Parses particle-count and resampling settings for the filter core."""
    defaults = TurtleBotParticleFilterConfig(particle_count=128, resample_threshold_ratio=0.5)
    return TurtleBotParticleFilterConfig(
        particle_count=int(raw.get("particle_count", defaults.particle_count)),
        resample_threshold_ratio=float(
            raw.get("resample_threshold_ratio", defaults.resample_threshold_ratio)
        ),
    )


def load_initial_pose_prior(raw: dict[str, Any]) -> Pose2DPrior:
    """Parses the initial localization prior used to seed particle initialization."""
    defaults = TurtleBotLocalizationConfig().initial_pose_prior
    mean_raw = raw.get("mean", {})
    return Pose2DPrior(
        mean=Pose2D(
            x=float(mean_raw.get("x", defaults.mean.x)),
            y=float(mean_raw.get("y", defaults.mean.y)),
            yaw=float(mean_raw.get("yaw", defaults.mean.yaw)),
        ),
        sigma_x=float(raw.get("sigma_x", defaults.sigma_x)),
        sigma_y=float(raw.get("sigma_y", defaults.sigma_y)),
        sigma_yaw=float(raw.get("sigma_yaw", defaults.sigma_yaw)),
    )


def load_motion_noise_settings(raw: dict[str, Any]) -> MotionNoiseSettings:
    """Parses motion-noise parameters applied during odometry prediction."""
    defaults = MotionNoiseSettings()
    return MotionNoiseSettings(
        x_meters=float(raw.get("x_meters", defaults.x_meters)),
        y_meters=float(raw.get("y_meters", defaults.y_meters)),
        yaw_radians=float(raw.get("yaw_radians", defaults.yaw_radians)),
    )


def load_measurement_settings(raw: dict[str, Any]) -> MeasurementSettings:
    """Parses measurement-scoring settings shared by live and offline evaluation."""
    defaults = MeasurementSettings()
    return MeasurementSettings(
        metric_name=raw.get("metric_name", defaults.metric_name),
        temperature=float(raw.get("temperature", defaults.temperature)),
        packed=bool(raw.get("packed", defaults.packed)),
        radius_clip=float(raw.get("radius_clip", defaults.radius_clip)),
        hybrid_ssim_weight=float(raw.get("hybrid_ssim_weight", defaults.hybrid_ssim_weight)),
        hybrid_l1_weight=float(raw.get("hybrid_l1_weight", defaults.hybrid_l1_weight)),
        hybrid_gradient_weight=float(raw.get("hybrid_gradient_weight", defaults.hybrid_gradient_weight)),
        lpips_top_k=int(raw.get("lpips_top_k", defaults.lpips_top_k)),
        lpips_weight=float(raw.get("lpips_weight", defaults.lpips_weight)),
        lpips_net=str(raw.get("lpips_net", defaults.lpips_net)),
    )


def load_initialization_settings(raw: dict[str, Any]) -> InitializationSettings:
    defaults = InitializationSettings()
    mode = str(raw.get("mode", defaults.mode)).strip().lower()
    if mode not in {"local", "global"}:
        raise ValueError(f"Unsupported initialization mode: {mode}")
    return InitializationSettings(
        mode=mode,
        global_yaw_uniform=bool(raw.get("global_yaw_uniform", defaults.global_yaw_uniform)),
    )


def load_recovery_settings(raw: dict[str, Any]) -> RecoverySettings:
    defaults = RecoverySettings()
    return RecoverySettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        alpha_slow=float(raw.get("alpha_slow", defaults.alpha_slow)),
        alpha_fast=float(raw.get("alpha_fast", defaults.alpha_fast)),
        random_particle_floor_ratio=float(
            raw.get("random_particle_floor_ratio", defaults.random_particle_floor_ratio)
        ),
        random_particle_max_ratio=float(
            raw.get("random_particle_max_ratio", defaults.random_particle_max_ratio)
        ),
    )
