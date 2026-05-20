"""Helpers for configuring and invoking LPIPS-based perceptual image scoring."""

from __future__ import annotations

from typing import Final

import lpips
import torch


_MODELS: dict[tuple[str, int | None, str], lpips.LPIPS] = {}
_SUPPORTED_NETS: Final[set[str]] = {"alex", "vgg", "squeeze"}


def get_lpips_model(device: torch.device, net: str) -> lpips.LPIPS:
    normalized_net = net.lower()
    if normalized_net not in _SUPPORTED_NETS:
        raise ValueError(f"Unsupported LPIPS net '{net}'. Supported: {sorted(_SUPPORTED_NETS)}")
    key = (device.type, device.index, normalized_net)
    model = _MODELS.get(key)
    if model is None:
        model = lpips.LPIPS(net=normalized_net).to(device)
        model.eval()
        _MODELS[key] = model
    return model


@torch.inference_mode()
def lpips_error(render_rgb: torch.Tensor, obs_rgb: torch.Tensor, *, device: torch.device, net: str) -> torch.Tensor:
    if render_rgb.ndim != 4 or render_rgb.shape[-1] != 3:
        raise ValueError("render_rgb must have shape [B,H,W,3]")
    if obs_rgb.ndim != 3 or obs_rgb.shape[-1] != 3:
        raise ValueError("obs_rgb must have shape [H,W,3]")
    model = get_lpips_model(device, net)
    render_nchw = render_rgb.permute(0, 3, 1, 2).mul(2.0).sub(1.0)
    obs_nchw = obs_rgb.permute(2, 0, 1)[None].expand(render_nchw.shape[0], -1, -1, -1).mul(2.0).sub(1.0)
    values = model(render_nchw, obs_nchw)
    return values.view(-1)
