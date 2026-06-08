"""Section-specific configuration parsers used by the top-level config loader."""

from __future__ import annotations

from typing import Any

from core.config.models import (
    AdaptiveParticleCountSettings,
    CameraOverrideSettings,
    InitializationSettings,
    MapSettings,
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


def load_map_settings(raw: dict[str, Any]) -> MapSettings:
    defaults = MapSettings()
    return MapSettings(
        yaml_path=str(raw.get("yaml_path", defaults.yaml_path)),
    )


def load_camera_override_settings(raw: dict[str, Any]) -> CameraOverrideSettings:
    defaults = CameraOverrideSettings()
    return CameraOverrideSettings(
        fx_scale=float(raw.get("fx_scale", defaults.fx_scale)),
        fy_scale=float(raw.get("fy_scale", defaults.fy_scale)),
        cx_offset=float(raw.get("cx_offset", defaults.cx_offset)),
        cy_offset=float(raw.get("cy_offset", defaults.cy_offset)),
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
    roughening_mode = str(raw.get("roughening_mode", defaults.roughening_mode)).strip().lower()
    if roughening_mode not in {"resample_only", "always"}:
        raise ValueError(f"Unsupported roughening mode: {roughening_mode}")
    return TurtleBotParticleFilterConfig(
        particle_count=int(raw.get("particle_count", defaults.particle_count)),
        resample_threshold_ratio=float(
            raw.get("resample_threshold_ratio", defaults.resample_threshold_ratio)
        ),
        roughening_enabled=bool(raw.get("roughening_enabled", defaults.roughening_enabled)),
        roughening_mode=roughening_mode,
        roughening_ratio=float(raw.get("roughening_ratio", defaults.roughening_ratio)),
        roughening_sigma_x=float(raw.get("roughening_sigma_x", defaults.roughening_sigma_x)),
        roughening_sigma_y=float(raw.get("roughening_sigma_y", defaults.roughening_sigma_y)),
        roughening_sigma_yaw=float(raw.get("roughening_sigma_yaw", defaults.roughening_sigma_yaw)),
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
    strategy = str(raw.get("strategy", defaults.strategy)).strip().lower()
    if strategy not in {"augmented_mcl", "absolute_score"}:
        raise ValueError(f"Unsupported recovery strategy: {strategy}")
    raw_profiles = raw.get("absolute_score_profiles", defaults.absolute_score_profiles)
    if not isinstance(raw_profiles, dict):
        raise ValueError("recovery.absolute_score_profiles must be a mapping keyed by metric name")
    absolute_score_profiles: dict[str, dict[str, float | int]] = {}
    for metric_name, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"recovery.absolute_score_profiles.{metric_name} must be a mapping")
        metric_key = str(metric_name).strip().lower()
        fallback = defaults.absolute_score_profiles["default"]
        absolute_score_profiles[metric_key] = {
            "best_score_threshold": float(
                profile.get("best_score_threshold", fallback["best_score_threshold"])
            ),
            "median_score_threshold": float(
                profile.get("median_score_threshold", fallback["median_score_threshold"])
            ),
            "random_particle_ratio": float(
                profile.get("random_particle_ratio", fallback["random_particle_ratio"])
            ),
            "consecutive_bad_updates": int(
                profile.get("consecutive_bad_updates", fallback["consecutive_bad_updates"])
            ),
        }
    if "default" not in absolute_score_profiles:
        absolute_score_profiles["default"] = defaults.absolute_score_profiles["default"].copy()
    return RecoverySettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        strategy=strategy,
        alpha_slow=float(raw.get("alpha_slow", defaults.alpha_slow)),
        alpha_fast=float(raw.get("alpha_fast", defaults.alpha_fast)),
        random_particle_floor_ratio=float(
            raw.get("random_particle_floor_ratio", defaults.random_particle_floor_ratio)
        ),
        random_particle_max_ratio=float(
            raw.get("random_particle_max_ratio", defaults.random_particle_max_ratio)
        ),
        absolute_score_profiles=absolute_score_profiles,
    )


def load_adaptive_particle_count_settings(raw: dict[str, Any]) -> AdaptiveParticleCountSettings:
    defaults = AdaptiveParticleCountSettings()
    max_particle_count = raw.get("max_particle_count", defaults.max_particle_count)
    settings = AdaptiveParticleCountSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        min_particle_count=int(raw.get("min_particle_count", defaults.min_particle_count)),
        medium_particle_count=int(raw.get("medium_particle_count", defaults.medium_particle_count)),
        max_particle_count=None if max_particle_count is None else int(max_particle_count),
        stable_required_updates=int(raw.get("stable_required_updates", defaults.stable_required_updates)),
        unstable_required_updates=int(raw.get("unstable_required_updates", defaults.unstable_required_updates)),
        xy_spread_stable_meters=float(raw.get("xy_spread_stable_meters", defaults.xy_spread_stable_meters)),
        xy_spread_unstable_meters=float(raw.get("xy_spread_unstable_meters", defaults.xy_spread_unstable_meters)),
        yaw_spread_stable_radians=float(raw.get("yaw_spread_stable_radians", defaults.yaw_spread_stable_radians)),
        yaw_spread_unstable_radians=float(raw.get("yaw_spread_unstable_radians", defaults.yaw_spread_unstable_radians)),
        best_score_stable_threshold=float(
            raw.get("best_score_stable_threshold", defaults.best_score_stable_threshold)
        ),
        median_score_stable_threshold=float(
            raw.get("median_score_stable_threshold", defaults.median_score_stable_threshold)
        ),
    )
    if settings.min_particle_count <= 0:
        raise ValueError("adaptive_particle_count.min_particle_count must be positive")
    if settings.medium_particle_count < settings.min_particle_count:
        raise ValueError("adaptive_particle_count.medium_particle_count must be >= min_particle_count")
    if settings.max_particle_count is not None and settings.max_particle_count < settings.medium_particle_count:
        raise ValueError("adaptive_particle_count.max_particle_count must be >= medium_particle_count")
    if settings.stable_required_updates <= 0:
        raise ValueError("adaptive_particle_count.stable_required_updates must be positive")
    if settings.unstable_required_updates <= 0:
        raise ValueError("adaptive_particle_count.unstable_required_updates must be positive")
    return settings
