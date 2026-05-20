"""Composes and runs the live localization backend entrypoint."""

from __future__ import annotations

import rclpy

from core.config import load_turtlebot_localization_config
from core.particle_filter.application.localization_service import TurtleBotLocalizationService
from core.particle_filter.infrastructure.renderer.renderer_service_client import RendererServiceClient
from core.particle_filter.infrastructure.ros.turtlebot_observation_source import TurtleBotObservationSource
from core.particle_filter.infrastructure.visualization.control_client import ControlCommandClient
from core.particle_filter.infrastructure.visualization.publisher import create_visualization_publisher


def main() -> None:
    settings = load_turtlebot_localization_config()

    rclpy.init()
    observation_source = TurtleBotObservationSource(settings.ros)
    renderer_client = RendererServiceClient(settings.renderer)
    visualization_publisher = create_visualization_publisher(settings.visualization)
    control_command_client = ControlCommandClient(settings.control)
    localization_service = TurtleBotLocalizationService(
        settings=settings,
        observation_source=observation_source,
        renderer_client=renderer_client,
        visualization_publisher=visualization_publisher,
        control_command_client=control_command_client,
    )

    try:
        localization_service.run()
    finally:
        localization_service.shutdown()
        observation_source.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
