"""Backward-compatible wrapper around the structured config package."""

from core.config import (
    DEFAULT_CONFIG_PATH,
    MeasurementSettings,
    MotionNoiseSettings,
    RuntimeSettings,
    TurtleBotLocalizationConfig,
    load_turtlebot_localization_config,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MeasurementSettings",
    "MotionNoiseSettings",
    "RuntimeSettings",
    "TurtleBotLocalizationConfig",
    "load_turtlebot_localization_config",
]
