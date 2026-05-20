"""Builds the active rendering backend from process-level environment settings."""

from __future__ import annotations

import os

from core.rendering.backends.base import RendererBackend


def create_renderer_backend() -> RendererBackend:
    """Creates the configured concrete rendering backend from environment settings."""
    backend_name = os.environ.get("RENDERER_BACKEND", "gsplat").strip().lower()
    if backend_name == "gsplat":
        from core.rendering.backends.gsplat import GsplatBackend

        return GsplatBackend()
    if backend_name == "vkdiff":
        from core.rendering.backends.vkdiff import VkdiffBackend

        return VkdiffBackend()
    raise ValueError(f"Unsupported renderer backend: {backend_name}")
