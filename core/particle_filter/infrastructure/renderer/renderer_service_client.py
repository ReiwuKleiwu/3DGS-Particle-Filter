"""HTTP client for the rendering service used during measurement updates."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image

from core.particle_filter.domain.pose import Pose2D
from core.particle_filter.infrastructure.ros.observation import CameraIntrinsics, TurtleBotObservation


@dataclass(frozen=True)
class RendererServiceSettings:
    backend: str = "gsplat"
    base_url: str = "http://127.0.0.1:8000"
    wait_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5
    score_batch_size: int = 16
    include_best_render_preview: bool = True


@dataclass(frozen=True)
class RendererScoreResult:
    errors: list[float]
    best_index: int
    elapsed_milliseconds: float
    best_render_png_bytes: bytes
    diagnostics: dict[str, Any] | None = None


class RendererServiceClient:
    def __init__(self, settings: RendererServiceSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> RendererServiceSettings:
        return self._settings

    def wait_until_ready(self) -> dict:
        """Polls the renderer health endpoint until the service reports that it is ready."""
        deadline = time.monotonic() + self._settings.wait_timeout_seconds
        last_error = None

        while time.monotonic() < deadline:
            try:
                status = self._request_json("/health", payload=None, timeout_seconds=5.0)
                if status.get("status") == "ok" and status.get("renderer_loaded"):
                    return status
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(self._settings.poll_interval_seconds)

        raise TimeoutError(
            f"Renderer service at {self._settings.base_url} was not ready within "
            f"{self._settings.wait_timeout_seconds:.1f}s. Last error: {last_error}"
        )

    def score_particles(
        self,
        *,
        particle_poses: list[Pose2D],
        observation: TurtleBotObservation,
        metric_name: str,
        packed: bool,
        radius_clip: float,
        hybrid_ssim_weight: float,
        hybrid_l1_weight: float,
        hybrid_gradient_weight: float,
        lpips_top_k: int,
        lpips_weight: float,
        lpips_net: str,
    ) -> RendererScoreResult:
        """Sends one scored particle batch to the renderer service and returns the scored result."""
        prep_start = time.perf_counter()
        observation_png_base64 = self._image_to_png_base64(observation.image_rgb)
        observation_encode_ms = (time.perf_counter() - prep_start) * 1000.0
        payload = {
            "poses": [self._pose_payload(pose) for pose in particle_poses],
            "camera": self._camera_payload(observation.camera),
            "observation_png_base64": observation_png_base64,
            "include_best_render_preview": self._settings.include_best_render_preview,
            "metric": metric_name,
            "packed": packed,
            "radius_clip": radius_clip,
            "hybrid_ssim_weight": hybrid_ssim_weight,
            "hybrid_l1_weight": hybrid_l1_weight,
            "hybrid_gradient_weight": hybrid_gradient_weight,
            "lpips_top_k": lpips_top_k,
            "lpips_weight": lpips_weight,
            "lpips_net": lpips_net,
            "max_batch_size": self._settings.score_batch_size,
        }

        request_start = time.perf_counter()
        response = self._request_json("/score_batch", payload=payload)
        roundtrip_ms = (time.perf_counter() - request_start) * 1000.0
        diagnostics = dict(response.get("diagnostics") or {})
        diagnostics["client_observation_png_encode_ms"] = observation_encode_ms
        diagnostics["client_http_roundtrip_ms"] = roundtrip_ms
        diagnostics["client_particle_count"] = len(particle_poses)
        return RendererScoreResult(
            errors=list(response["scores"]),
            best_index=int(response["best_index"]),
            elapsed_milliseconds=float(response["elapsed_ms"]),
            best_render_png_bytes=base64.b64decode(response["best_render_png_base64"]),
            diagnostics=diagnostics,
        )

    def _request_json(self, path: str, *, payload: dict | None, timeout_seconds: float | None = None) -> dict:
        """Performs a JSON HTTP request against the renderer service and decodes the response."""
        request_timeout_seconds = timeout_seconds or self._settings.request_timeout_seconds
        request_url = f"{self._settings.base_url.rstrip('/')}{path}"
        request_headers = {"Accept": "application/json"}
        request_data = None

        if payload is not None:
            request_headers["Content-Type"] = "application/json"
            request_data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(request_url, data=request_data, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Renderer request failed with HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _pose_payload(pose: Pose2D) -> dict:
        return {"x": pose.x, "y": pose.y, "yaw": pose.yaw}

    @staticmethod
    def _camera_payload(camera: CameraIntrinsics) -> dict:
        return {
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        }

    @staticmethod
    def _image_to_png_base64(image_rgb) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image_rgb, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
