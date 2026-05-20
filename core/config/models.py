"""Configuration dataclasses shared across backend subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.particle_filter.domain.particle_filter import TurtleBotParticleFilterConfig
from core.particle_filter.domain.pose import Pose2D, Pose2DPrior
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceSettings
from core.particle_filter.infrastructure.ros.observation import RosTopicSettings
from core.particle_filter.infrastructure.visualization.control_client import ControlCommandClientSettings
from core.particle_filter.infrastructure.visualization.publisher import VisualizationPublisherSettings


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
    lpips_top_k: int = 0
    lpips_weight: float = 0.0
    lpips_net: str = "alex"


@dataclass(frozen=True)
class RuntimeSettings:
    observation_ready_timeout_seconds: float = 10.0
    spin_timeout_seconds: float = 0.05
    random_seed: int | None = None


@dataclass(frozen=True)
class TurtleBotLocalizationConfig:
    renderer: RendererServiceSettings = field(default_factory=RendererServiceSettings)
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
