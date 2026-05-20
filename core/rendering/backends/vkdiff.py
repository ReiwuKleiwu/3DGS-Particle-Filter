"""Concrete rendering backend that manages the native VkDiff forward-render subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from core.rendering.backends.base import RendererBackend
from core.rendering.backends.vkdiff_scene import camera_entry, convert_point_cloud_to_vkdiff
from core.rendering.types import CameraSpec, Pose2D
from core.rendering.scoring import ScoreMetric, image_file_to_tensor, tensor_to_png_bytes


class VkdiffBackend(RendererBackend):
    backend_name = "vkdiff"

    def __init__(self, *, ply_path: str | Path | None = None) -> None:
        self._splat_path = Path(ply_path or os.environ.get("SPLAT_PATH", "/workspace/splat.ply")).resolve()
        self._device = torch.device("cuda")
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="vkdiff-backend-"))
        self._converted_ply = self._runtime_dir / "point_cloud.ply"
        self._gaussian_count = convert_point_cloud_to_vkdiff(self._splat_path, self._converted_ply)
        self._server_path = Path(
            os.environ.get("VKDIFF_SERVER_PATH", "/opt/vkdiff_build/ForwardRenderServer")
        )
        if not self._server_path.is_file():
            raise FileNotFoundError(f"VkDiff server binary not found: {self._server_path}")
        self._server = subprocess.Popen(
            [str(self._server_path), str(self._converted_ply)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._wait_for_ready()

    @property
    def splat_path(self) -> Path:
        return self._splat_path

    @property
    def gaussian_count(self) -> int:
        return self._gaussian_count

    @property
    def device(self) -> torch.device:
        return self._device

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
        """Renders a batch of poses by delegating to the native VkDiff worker process."""
        if not poses:
            raise ValueError("At least one pose is required")
        if packed is not None or radius_clip is not None or sh_degree is not None or max_batch_size is not None:
            # Keep the API stable, but be explicit that these knobs do not apply to VkDiff.
            pass

        build_start = time.perf_counter()
        entries = []
        for index, pose in enumerate(poses):
            dataset_entry = camera_entry(pose, camera, f"img_{index:04d}")
            entries.append(
                {
                    "image_name": dataset_entry["img_name"],
                    "width": dataset_entry["width"],
                    "height": dataset_entry["height"],
                    "fx": dataset_entry["fx"],
                    "fy": dataset_entry["fy"],
                    "position": dataset_entry["position"],
                    "rotation": dataset_entry["rotation"],
                }
            )
        payload_build_ms = (time.perf_counter() - build_start) * 1000.0
        payload = {"entries": entries}
        request_start = time.perf_counter()
        response, pixel_bytes = self._request(payload)
        request_roundtrip_ms = (time.perf_counter() - request_start) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(f"VkDiff render request failed: {response.get('error', 'unknown error')}")

        decode_start = time.perf_counter()
        pixels = np.frombuffer(pixel_bytes, dtype=np.float32)
        expected = len(poses) * camera.width * camera.height * 3
        if pixels.size != expected:
            raise RuntimeError(
                f"Unexpected VkDiff output size: got {pixels.size} floats, expected {expected}"
            )
        cpu_tensor = torch.from_numpy(
            pixels.reshape(len(poses), 3, camera.height, camera.width).transpose(0, 2, 3, 1).copy()
        )
        output_decode_ms = (time.perf_counter() - decode_start) * 1000.0
        stack_start = time.perf_counter()
        result = cpu_tensor.to(self._device)
        stack_ms = (time.perf_counter() - stack_start) * 1000.0
        self._last_render_diagnostics = {
            "backend": self.backend_name,
            "pose_count": len(poses),
            "payload_build_ms": payload_build_ms,
            "server_elapsed_ms": float(response.get("elapsed_ms", 0.0)),
            "request_roundtrip_ms": request_roundtrip_ms,
            "output_decode_ms": output_decode_ms,
            "device_upload_ms": stack_ms,
            "render_ms": payload_build_ms + request_roundtrip_ms + output_decode_ms + stack_ms,
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
        """Renders and scores a pose batch through the native VkDiff scoring path."""
        if not poses:
            raise ValueError("At least one pose is required")
        if packed is not None or radius_clip is not None or sh_degree is not None or max_batch_size is not None:
            pass

        obs_decode_start = time.perf_counter()
        obs_tensor = image_file_to_tensor(observation_png_bytes, camera.width, camera.height, torch.device("cpu"))
        obs_decode_ms = (time.perf_counter() - obs_decode_start) * 1000.0
        observation_bytes = obs_tensor.contiguous().numpy().astype(np.float32, copy=False).tobytes()

        build_start = time.perf_counter()
        entries = []
        for index, pose in enumerate(poses):
            dataset_entry = camera_entry(pose, camera, f"img_{index:04d}")
            entries.append(
                {
                    "image_name": dataset_entry["img_name"],
                    "width": dataset_entry["width"],
                    "height": dataset_entry["height"],
                    "fx": dataset_entry["fx"],
                    "fy": dataset_entry["fy"],
                    "position": dataset_entry["position"],
                    "rotation": dataset_entry["rotation"],
                }
            )
        payload_build_ms = (time.perf_counter() - build_start) * 1000.0
        payload = {
            "operation": "score",
            "entries": entries,
            "observation_bytes": len(observation_bytes),
            "include_best_render_preview": include_best_render_preview,
            "metric": metric,
            "lpips_net": lpips_net,
            "ssim_window_size": ssim_window_size,
            "hybrid_ssim_weight": hybrid_ssim_weight,
            "hybrid_l1_weight": hybrid_l1_weight,
            "hybrid_gradient_weight": hybrid_gradient_weight,
        }
        request_start = time.perf_counter()
        response, best_image_bytes = self._request(payload, binary_request=observation_bytes)
        request_roundtrip_ms = (time.perf_counter() - request_start) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(f"VkDiff score request failed: {response.get('error', 'unknown error')}")

        best_png_encode_ms = 0.0
        best_render_png_bytes = b""
        if best_image_bytes:
            best_decode_start = time.perf_counter()
            best_pixels = np.frombuffer(best_image_bytes, dtype=np.float32)
            expected = camera.width * camera.height * 3
            if best_pixels.size != expected:
                raise RuntimeError(
                    f"Unexpected VkDiff best image size: got {best_pixels.size} floats, expected {expected}"
                )
            best_tensor = torch.from_numpy(
                best_pixels.reshape(3, camera.height, camera.width).transpose(1, 2, 0).copy()
            )
            best_render_png_bytes = tensor_to_png_bytes(best_tensor)
            best_png_encode_ms = (time.perf_counter() - best_decode_start) * 1000.0

        diagnostics = {
            "backend": self.backend_name,
            "pose_count": len(poses),
            "observation_decode_ms": obs_decode_ms,
            "payload_build_ms": payload_build_ms,
            "server_elapsed_ms": float(response.get("elapsed_ms", 0.0)),
            "render_elapsed_ms": float(response.get("render_elapsed_ms", response.get("elapsed_ms", 0.0))),
            "render_wall_ms": float(response.get("render_wall_ms", 0.0)),
            "render_submit_overhead_ms": float(response.get("render_submit_overhead_ms", 0.0)),
            "score_gpu_ms": float(response.get("score_gpu_ms", 0.0)),
            "score_sync_ms": float(response.get("score_sync_ms", 0.0)),
            "observation_upload_ms": float(response.get("observation_upload_ms", 0.0)),
            "observation_preprocess_ms": float(response.get("observation_preprocess_ms", 0.0)),
            "best_copy_d2d_ms": float(response.get("best_copy_d2d_ms", 0.0)),
            "best_preview_d2h_ms": float(response.get("best_preview_d2h_ms", 0.0)),
            "worker_total_ms": float(response.get("worker_total_ms", 0.0)),
            "worker_residual_ms": float(response.get("worker_residual_ms", 0.0)),
            "request_roundtrip_ms": request_roundtrip_ms,
            "best_png_encode_ms": best_png_encode_ms,
        }
        self._last_render_diagnostics = diagnostics
        return {
            "scores": list(response["scores"]),
            "best_index": int(response["best_index"]),
            "best_render_png_bytes": best_render_png_bytes,
            "diagnostics": diagnostics,
        }

    def close(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            try:
                if server.stdin is not None:
                    server.stdin.close()
                server.terminate()
                server.wait(timeout=5.0)
            except Exception:
                server.kill()
            finally:
                self._server = None
        runtime_dir = getattr(self, "_runtime_dir", None)
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    def __del__(self) -> None:
        self.close()

    def get_last_render_diagnostics(self) -> dict[str, float | int | str | bool] | None:
        return getattr(self, "_last_render_diagnostics", None)

    def _wait_for_ready(self) -> None:
        if self._server.stderr is None:
            raise RuntimeError("VkDiff server stderr pipe is unavailable")
        while True:
            line = self._server.stderr.readline()
            if line == b"":
                raise RuntimeError("VkDiff server exited before becoming ready")
            if b"ForwardRenderServer ready" in line:
                return

    def _request(self, payload: dict, binary_request: bytes = b"") -> tuple[dict, bytes]:
        if self._server.stdin is None or self._server.stdout is None:
            raise RuntimeError("VkDiff server pipes are unavailable")
        self._server.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        if binary_request:
            self._server.stdin.write(binary_request)
        self._server.stdin.flush()
        response_line = self._server.stdout.readline()
        if response_line == b"":
            stderr = ""
            if self._server.stderr is not None:
                stderr = self._server.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VkDiff server exited unexpectedly. stderr={stderr}")
        response = json.loads(response_line.decode("utf-8"))
        payload_bytes = int(response.get("payload_bytes", 0))
        binary_payload = b""
        if payload_bytes:
            chunks: list[bytes] = []
            remaining = payload_bytes
            while remaining > 0:
                chunk = self._server.stdout.read(remaining)
                if chunk == b"":
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            binary_payload = b"".join(chunks)
            if len(binary_payload) != payload_bytes:
                raise RuntimeError(
                    f"VkDiff server returned incomplete payload: got {len(binary_payload)} bytes, expected {payload_bytes}"
                )
        return response, binary_payload
