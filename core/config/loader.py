"""Top-level composition loader for the backend runtime configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config.models import TurtleBotLocalizationConfig
from core.config.sections import (
    load_control_settings,
    load_initial_pose_prior,
    load_measurement_settings,
    load_motion_noise_settings,
    load_particle_filter_settings,
    load_renderer_settings,
    load_ros_topic_settings,
    load_runtime_settings,
    load_visualization_settings,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "turtlebot_localization.yaml"


def load_turtlebot_localization_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> TurtleBotLocalizationConfig:
    """Loads the YAML config file and composes the full backend runtime settings object."""
    resolved_path = Path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    return TurtleBotLocalizationConfig(
        renderer=load_renderer_settings(raw_config.get("renderer", {})),
        ros=load_ros_topic_settings(raw_config.get("ros", {})),
        runtime=load_runtime_settings(raw_config.get("runtime", {})),
        visualization=load_visualization_settings(raw_config.get("visualization", {})),
        control=load_control_settings(raw_config.get("control", raw_config.get("reset_control", {}))),
        particle_filter=load_particle_filter_settings(raw_config.get("particle_filter", {})),
        initial_pose_prior=load_initial_pose_prior(raw_config.get("initial_pose_prior", {})),
        motion_noise=load_motion_noise_settings(raw_config.get("motion_noise", {})),
        measurement=load_measurement_settings(raw_config.get("measurement", {})),
    )
