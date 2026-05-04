from __future__ import annotations

import io
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ScoreMetric = Literal["mse", "ssim", "rgb-ssim", "hybrid"]


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _ssim_error_nchw(x: torch.Tensor, y: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError("ssim window size must be odd")
    padding = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.mean(dim=(1, 2, 3))


def rgb_ssim_error(render_rgb: torch.Tensor, obs_rgb: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    x = render_rgb.permute(0, 3, 1, 2)
    y = obs_rgb.permute(2, 0, 1)[None].expand_as(x)
    return _ssim_error_nchw(x, y, window_size)


def rgb_l1_error(render_rgb: torch.Tensor, obs_rgb: torch.Tensor) -> torch.Tensor:
    return torch.abs(render_rgb - obs_rgb[None]).mean(dim=(1, 2, 3))


def gradient_l1_error(render_gray: torch.Tensor, obs_gray: torch.Tensor) -> torch.Tensor:
    render_dx = render_gray[:, :, 1:] - render_gray[:, :, :-1]
    render_dy = render_gray[:, 1:, :] - render_gray[:, :-1, :]
    obs_dx = obs_gray[:, 1:] - obs_gray[:, :-1]
    obs_dy = obs_gray[1:, :] - obs_gray[:-1, :]
    dx_error = torch.abs(render_dx - obs_dx[None]).mean(dim=(1, 2))
    dy_error = torch.abs(render_dy - obs_dy[None]).mean(dim=(1, 2))
    return 0.5 * (dx_error + dy_error)


def image_error(
    render_rgb: torch.Tensor,
    obs_rgb: torch.Tensor,
    render_gray: torch.Tensor,
    obs_gray: torch.Tensor,
    metric: ScoreMetric,
    ssim_window_size: int,
    hybrid_ssim_weight: float,
    hybrid_l1_weight: float,
    hybrid_gradient_weight: float,
) -> torch.Tensor:
    if metric == "mse":
        return ((render_gray - obs_gray[None]) ** 2).mean(dim=(1, 2))
    if metric == "ssim":
        ref = obs_gray[None, None, :, :].expand(render_gray.shape[0], -1, -1, -1)
        return _ssim_error_nchw(render_gray[:, None, :, :], ref, ssim_window_size)
    if metric == "rgb-ssim":
        return rgb_ssim_error(render_rgb, obs_rgb, ssim_window_size)
    if metric == "hybrid":
        total_weight = hybrid_ssim_weight + hybrid_l1_weight + hybrid_gradient_weight
        if total_weight <= 0.0:
            raise ValueError("At least one hybrid score weight must be positive")
        return (
            hybrid_ssim_weight * rgb_ssim_error(render_rgb, obs_rgb, ssim_window_size)
            + hybrid_l1_weight * rgb_l1_error(render_rgb, obs_rgb)
            + hybrid_gradient_weight * gradient_l1_error(render_gray, obs_gray)
        ) / total_weight
    raise ValueError(f"Unsupported score metric: {metric}")


def image_file_to_tensor(image_file: bytes, width: int, height: int, device: torch.device) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def tensor_to_png_bytes(rgb: torch.Tensor) -> bytes:
    image = rgb.detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
