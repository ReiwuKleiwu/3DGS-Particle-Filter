from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from particle_filter.domain.particle_filter import TurtleBotParticleFilterConfig
from particle_filter.domain.pose import Pose2D, Pose2DPrior
from particle_filter.infrastructure.renderer.splat_renderer_client import RendererClientSettings
from particle_filter.infrastructure.ros.observation import RosTopicSettings
from particle_filter.infrastructure.visualization.control_client import ControlCommandClientSettings
from particle_filter.infrastructure.visualization.publisher import VisualizationPublisherSettings


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "turtlebot_localization.yaml"


@dataclass(frozen=True)
class MotionNoiseSettings:
    x_meters: float = 0.02
    y_meters: float = 0.02
    yaw_radians: float = 0.017453292519943295


@dataclass(frozen=True)
class MeasurementSettings:
    metric_name: str = "hybrid"
    temperature: float = 0.02
    packed: bool = False
    radius_clip: float = 3.0
    hybrid_ssim_weight: float = 0.50
    hybrid_l1_weight: float = 0.25
    hybrid_gradient_weight: float = 0.25


@dataclass(frozen=True)
class RuntimeSettings:
    observation_ready_timeout_seconds: float = 10.0
    spin_timeout_seconds: float = 0.05


@dataclass(frozen=True)
class TurtleBotLocalizationConfig:
    renderer: RendererClientSettings = field(default_factory=RendererClientSettings)
    ros: RosTopicSettings = field(default_factory=RosTopicSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    visualization: VisualizationPublisherSettings = field(default_factory=VisualizationPublisherSettings)
    control: ControlCommandClientSettings = field(default_factory=ControlCommandClientSettings)
    particle_filter: TurtleBotParticleFilterConfig = field(
        default_factory=lambda: TurtleBotParticleFilterConfig(
            particle_count=128,
            resample_threshold_ratio=0.5,
        )
    )
    initial_pose_prior: Pose2DPrior = field(
        default_factory=lambda: Pose2DPrior(
            mean=Pose2D(x=-2.685, y=-2.003, yaw=-0.020),
            sigma_x=0.5,
            sigma_y=0.5,
            sigma_yaw=0.5,
        )
    )
    motion_noise: MotionNoiseSettings = field(default_factory=MotionNoiseSettings)
    measurement: MeasurementSettings = field(default_factory=MeasurementSettings)


def load_turtlebot_localization_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> TurtleBotLocalizationConfig:
    resolved_path = Path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    return TurtleBotLocalizationConfig(
        renderer=_load_renderer_settings(raw_config.get("renderer", {})),
        ros=_load_ros_topic_settings(raw_config.get("ros", {})),
        runtime=_load_runtime_settings(raw_config.get("runtime", {})),
        visualization=_load_visualization_settings(raw_config.get("visualization", {})),
        control=_load_control_settings(raw_config.get("control", raw_config.get("reset_control", {}))),
        particle_filter=_load_particle_filter_settings(raw_config.get("particle_filter", {})),
        initial_pose_prior=_load_initial_pose_prior(raw_config.get("initial_pose_prior", {})),
        motion_noise=_load_motion_noise_settings(raw_config.get("motion_noise", {})),
        measurement=_load_measurement_settings(raw_config.get("measurement", {})),
    )


def _load_renderer_settings(raw: dict[str, Any]) -> RendererClientSettings:
    defaults = RendererClientSettings()
    return RendererClientSettings(
        base_url=raw.get("base_url", defaults.base_url),
        wait_timeout_seconds=float(raw.get("wait_timeout_seconds", defaults.wait_timeout_seconds)),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
        poll_interval_seconds=float(raw.get("poll_interval_seconds", defaults.poll_interval_seconds)),
        score_batch_size=int(raw.get("score_batch_size", defaults.score_batch_size)),
    )


def _load_ros_topic_settings(raw: dict[str, Any]) -> RosTopicSettings:
    defaults = RosTopicSettings()
    return RosTopicSettings(
        image_topic=raw.get("image_topic", defaults.image_topic),
        camera_info_topic=raw.get("camera_info_topic", defaults.camera_info_topic),
        odometry_topic=raw.get("odometry_topic", defaults.odometry_topic),
        map_frame=raw.get("map_frame", defaults.map_frame),
        base_frame=raw.get("base_frame", defaults.base_frame),
        tf_lookup_mode=raw.get("tf_lookup_mode", defaults.tf_lookup_mode),
        tf_timeout_seconds=float(raw.get("tf_timeout_seconds", defaults.tf_timeout_seconds)),
        require_odometry=bool(raw.get("require_odometry", defaults.require_odometry)),
    )


def _load_runtime_settings(raw: dict[str, Any]) -> RuntimeSettings:
    defaults = RuntimeSettings()
    return RuntimeSettings(
        observation_ready_timeout_seconds=float(
            raw.get("observation_ready_timeout_seconds", defaults.observation_ready_timeout_seconds)
        ),
        spin_timeout_seconds=float(raw.get("spin_timeout_seconds", defaults.spin_timeout_seconds)),
    )


def _load_visualization_settings(raw: dict[str, Any]) -> VisualizationPublisherSettings:
    defaults = VisualizationPublisherSettings()
    return VisualizationPublisherSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        publish_url=raw.get("publish_url", defaults.publish_url),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
        observation_jpeg_quality=int(raw.get("observation_jpeg_quality", defaults.observation_jpeg_quality)),
    )


def _load_control_settings(raw: dict[str, Any]) -> ControlCommandClientSettings:
    defaults = ControlCommandClientSettings()
    return ControlCommandClientSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        poll_url=raw.get("poll_url", defaults.poll_url),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", defaults.request_timeout_seconds)),
    )


def _load_particle_filter_settings(raw: dict[str, Any]) -> TurtleBotParticleFilterConfig:
    defaults = TurtleBotParticleFilterConfig(particle_count=128, resample_threshold_ratio=0.5)
    return TurtleBotParticleFilterConfig(
        particle_count=int(raw.get("particle_count", defaults.particle_count)),
        resample_threshold_ratio=float(
            raw.get("resample_threshold_ratio", defaults.resample_threshold_ratio)
        ),
    )


def _load_initial_pose_prior(raw: dict[str, Any]) -> Pose2DPrior:
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


def _load_motion_noise_settings(raw: dict[str, Any]) -> MotionNoiseSettings:
    defaults = MotionNoiseSettings()
    return MotionNoiseSettings(
        x_meters=float(raw.get("x_meters", defaults.x_meters)),
        y_meters=float(raw.get("y_meters", defaults.y_meters)),
        yaw_radians=float(raw.get("yaw_radians", defaults.yaw_radians)),
    )


def _load_measurement_settings(raw: dict[str, Any]) -> MeasurementSettings:
    defaults = MeasurementSettings()
    return MeasurementSettings(
        metric_name=raw.get("metric_name", defaults.metric_name),
        temperature=float(raw.get("temperature", defaults.temperature)),
        packed=bool(raw.get("packed", defaults.packed)),
        radius_clip=float(raw.get("radius_clip", defaults.radius_clip)),
        hybrid_ssim_weight=float(raw.get("hybrid_ssim_weight", defaults.hybrid_ssim_weight)),
        hybrid_l1_weight=float(raw.get("hybrid_l1_weight", defaults.hybrid_l1_weight)),
        hybrid_gradient_weight=float(raw.get("hybrid_gradient_weight", defaults.hybrid_gradient_weight)),
    )
