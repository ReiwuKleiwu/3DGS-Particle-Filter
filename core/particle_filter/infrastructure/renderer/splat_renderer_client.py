"""Backward-compatible aliases for older imports that still refer to the splat renderer."""

from core.particle_filter.infrastructure.renderer.renderer_service_client import (
    RendererScoreResult,
    RendererServiceClient as SplatRendererClient,
    RendererServiceSettings as RendererClientSettings,
)

__all__ = ["RendererClientSettings", "RendererScoreResult", "SplatRendererClient"]
