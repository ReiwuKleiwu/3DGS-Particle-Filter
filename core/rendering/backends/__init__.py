"""Common exports for selecting and typing concrete rendering backend implementations."""

from core.rendering.backends.base import RendererBackend
from core.rendering.backends.factory import create_renderer_backend

__all__ = ["RendererBackend", "create_renderer_backend"]
