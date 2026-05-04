from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image

from particle_filter.domain.pose import Pose2D
from particle_filter.infrastructure.ros.observation import CameraIntrinsics, TurtleBotObservation


@dataclass(frozen=True)
class RendererClientSettings:
    base_url: str = "http://127.0.0.1:8000"
    wait_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5
    score_batch_size: int = 16


@dataclass(frozen=True)
class RendererScoreResult:
    errors: list[float]
    best_index: int
    elapsed_milliseconds: float
    best_render_png_bytes: bytes


class SplatRendererClient:
    def __init__(self, settings: RendererClientSettings) -> None:
        self._settings = settings

    def wait_until_ready(self) -> dict:
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
    ) -> RendererScoreResult:
        payload = {
            "poses": [self._pose_payload(pose) for pose in particle_poses],
            "camera": self._camera_payload(observation.camera),
            "observation_png_base64": self._image_to_png_base64(observation.image_rgb),
            "metric": metric_name,
            "packed": packed,
            "radius_clip": radius_clip,
            "hybrid_ssim_weight": hybrid_ssim_weight,
            "hybrid_l1_weight": hybrid_l1_weight,
            "hybrid_gradient_weight": hybrid_gradient_weight,
            "max_batch_size": self._settings.score_batch_size,
        }

        response = self._request_json("/score_batch", payload=payload)
        return RendererScoreResult(
            errors=list(response["scores"]),
            best_index=int(response["best_index"]),
            elapsed_milliseconds=float(response["elapsed_ms"]),
            best_render_png_bytes=base64.b64decode(response["best_render_png_base64"]),
        )

    def _request_json(self, path: str, *, payload: dict | None, timeout_seconds: float | None = None) -> dict:
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
