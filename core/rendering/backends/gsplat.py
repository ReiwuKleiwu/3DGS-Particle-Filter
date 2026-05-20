"""Concrete rendering backend that wraps the in-process Python gsplat implementation."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from pytorch_msssim import ssim as msssim_ssim

from core.rendering.backends.base import RendererBackend
from core.rendering.lpips import lpips_error
from core.rendering.types import CameraSpec, Pose2D
from core.rendering.backends.gsplat_core import SplatRenderer
from core.rendering.scoring import (
    ScoreMetric,
    image_file_to_tensor,
    gradient_l1_error,
    rgb_l1_error,
    rgb_to_gray,
    tensor_to_png_bytes,
)


def _image_error_fast(
    render_rgb: torch.Tensor,
    obs_rgb: torch.Tensor,
    render_gray: torch.Tensor,
    obs_gray: torch.Tensor,
    metric: ScoreMetric,
    lpips_net: str,
    device: torch.device,
    ssim_window_size: int,
    hybrid_ssim_weight: float,
    hybrid_l1_weight: float,
    hybrid_gradient_weight: float,
) -> torch.Tensor:
    if metric == "mse":
        return ((render_gray - obs_gray[None]) ** 2).mean(dim=(1, 2))

    if metric == "ssim":
        x = render_gray[:, None, :, :]
        y = obs_gray[None, None, :, :].expand_as(x)
        return 1.0 - msssim_ssim(x, y, data_range=1.0, size_average=False)

    if metric == "rgb-ssim":
        x = render_rgb.permute(0, 3, 1, 2)
        y = obs_rgb.permute(2, 0, 1)[None].expand_as(x)
        return 1.0 - msssim_ssim(x, y, data_range=1.0, size_average=False)

    if metric == "hybrid":
        total_weight = hybrid_ssim_weight + hybrid_l1_weight + hybrid_gradient_weight
        if total_weight <= 0.0:
            raise ValueError("At least one hybrid score weight must be positive")
        x = render_rgb.permute(0, 3, 1, 2)
        y = obs_rgb.permute(2, 0, 1)[None].expand_as(x)
        ssim_term = 1.0 - msssim_ssim(x, y, data_range=1.0, size_average=False)
        return (
            hybrid_ssim_weight * ssim_term
            + hybrid_l1_weight * rgb_l1_error(render_rgb, obs_rgb)
            + hybrid_gradient_weight * gradient_l1_error(render_gray, obs_gray)
        ) / total_weight

    if metric == "lpips":
        return lpips_error(render_rgb, obs_rgb, device=device, net=lpips_net)

    raise ValueError(f"Unsupported score metric: {metric}")


class GsplatBackend(RendererBackend):
    backend_name = "gsplat"

    def __init__(self, *, ply_path: str | Path | None = None) -> None:
        self._renderer = SplatRenderer(ply_path=ply_path)

    @property
    def splat_path(self) -> Path:
        return self._renderer.ply_path

    @property
    def gaussian_count(self) -> int:
        return int(self._renderer.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self._renderer.device

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
        """Renders a batch of poses with the in-process gsplat renderer."""
        start = time.perf_counter()
        result = self._renderer.render_batch_chunked(
            poses=poses,
            camera=camera,
            packed=packed,
            white_background=white_background,
            radius_clip=radius_clip,
            sh_degree=sh_degree,
            max_batch_size=max_batch_size,
        )
        self._last_render_diagnostics = {
            "backend": self.backend_name,
            "pose_count": len(poses),
            "render_ms": (time.perf_counter() - start) * 1000.0,
        }
        return result

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
        """Renders a pose batch and scores it against the observation using native tensor ops."""
        if not poses:
            raise ValueError("At least one pose is required")

        observation_decode_start = time.perf_counter()
        obs_rgb = image_file_to_tensor(observation_png_bytes, camera.width, camera.height, self.device)
        observation_decode_ms = (time.perf_counter() - observation_decode_start) * 1000.0

        render_start = time.perf_counter()
        render_rgb = self._renderer.render_batch_chunked(
            poses=poses,
            camera=camera,
            packed=packed,
            white_background=False,
            radius_clip=radius_clip,
            sh_degree=sh_degree,
            max_batch_size=max_batch_size,
        )
        render_elapsed_ms = (time.perf_counter() - render_start) * 1000.0

        gray_start = time.perf_counter()
        render_gray = rgb_to_gray(render_rgb)
        obs_gray = rgb_to_gray(obs_rgb)
        grayscale_ms = (time.perf_counter() - gray_start) * 1000.0

        score_start = time.perf_counter()
        errors = _image_error_fast(
            render_rgb,
            obs_rgb,
            render_gray,
            obs_gray,
            metric,
            lpips_net,
            self.device,
            ssim_window_size,
            hybrid_ssim_weight,
            hybrid_l1_weight,
            hybrid_gradient_weight,
        )
        score_gpu_ms = (time.perf_counter() - score_start) * 1000.0
        best_index = int(errors.argmin().item())

        best_png_encode_ms = 0.0
        best_render_png_bytes = b""
        if include_best_render_preview:
            best_png_start = time.perf_counter()
            best_render_png_bytes = tensor_to_png_bytes(render_rgb[best_index])
            best_png_encode_ms = (time.perf_counter() - best_png_start) * 1000.0

        diagnostics = {
            "backend": self.backend_name,
            "pose_count": len(poses),
            "observation_decode_ms": observation_decode_ms,
            "render_elapsed_ms": render_elapsed_ms,
            "score_gpu_ms": score_gpu_ms,
            "grayscale_ms": grayscale_ms,
            "best_png_encode_ms": best_png_encode_ms,
            "worker_total_ms": observation_decode_ms + render_elapsed_ms + grayscale_ms + score_gpu_ms + best_png_encode_ms,
            "server_elapsed_ms": render_elapsed_ms,
        }
        self._last_render_diagnostics = diagnostics
        return {
            "scores": errors.detach().cpu().tolist(),
            "best_index": best_index,
            "best_render_png_bytes": best_render_png_bytes,
            "diagnostics": diagnostics,
        }

    def get_last_render_diagnostics(self) -> dict[str, float | int | str | bool] | None:
        return getattr(self, "_last_render_diagnostics", None)
