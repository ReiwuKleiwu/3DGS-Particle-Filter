"""Exports LPIPS models to ONNX for use by the native VkDiff backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import lpips
import torch


class LpipsOnnxWrapper(torch.nn.Module):
    def __init__(self, net: str) -> None:
        super().__init__()
        self.metric = lpips.LPIPS(net=net)
        self.metric.eval()

    def forward(self, render: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        render = render * 2.0 - 1.0
        obs = obs * 2.0 - 1.0
        values = self.metric(render, obs)
        return values.view(-1)


def export_model(*, net: str, width: int, height: int, output_path: Path) -> None:
    model = LpipsOnnxWrapper(net).eval()
    render = torch.rand(1, 3, height, width, dtype=torch.float32)
    obs = torch.rand(1, 3, height, width, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (render, obs),
        str(output_path),
        input_names=["render", "obs"],
        output_names=["score"],
        dynamic_axes={
            "render": {0: "batch"},
            "obs": {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", choices=["alex", "vgg", "squeeze"], required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_model(net=args.net, width=args.width, height=args.height, output_path=args.output)


if __name__ == "__main__":
    main()
