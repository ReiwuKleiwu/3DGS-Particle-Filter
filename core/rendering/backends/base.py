"""Abstract contract implemented by gsplat and VkDiff rendering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from core.rendering.types import CameraSpec, Pose2D
from core.rendering.scoring import ScoreMetric


class RendererBackend(ABC):
    backend_name: str

    @property
    @abstractmethod
    def splat_path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def gaussian_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def render_batch(
        self,
        *,
        poses: list[Pose2D],
        camera: CameraSpec,
        packed: bool | None = None,
        white_background: bool = False,
        radius_clip: float | None = None,
        sh_degree: int | None = None,
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_last_render_diagnostics(self) -> dict[str, float | int | str | bool] | None:
        return None

    def score_batch_native(
        self,
        *,
        poses: list[Pose2D],
        camera: CameraSpec,
        observation_png_bytes: bytes,
        metric: ScoreMetric,
        lpips_net: str = "alex",
        include_best_render_preview: bool,
        ssim_window_size: int,
        hybrid_ssim_weight: float,
        hybrid_l1_weight: float,
        hybrid_gradient_weight: float,
        packed: bool | None = None,
        radius_clip: float | None = None,
        sh_degree: int | None = None,
        max_batch_size: int | None = None,
    ) -> dict | None:
        return None
