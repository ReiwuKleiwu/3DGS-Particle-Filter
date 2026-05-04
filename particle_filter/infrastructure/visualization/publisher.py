from __future__ import annotations

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from particle_filter.infrastructure.visualization.models import VisualizationSnapshot


@dataclass(frozen=True)
class VisualizationPublisherSettings:
    enabled: bool = False
    publish_url: str = "http://127.0.0.1:8090/api/publish-latest"
    request_timeout_seconds: float = 0.25
    observation_jpeg_quality: int = 80


class VisualizationPublisher(Protocol):
    def publish(self, snapshot: VisualizationSnapshot) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpVisualizationPublisher:
    def publish(self, snapshot: VisualizationSnapshot) -> None:  # noqa: ARG002
        return

    def close(self) -> None:
        return


class LatestOnlyHttpVisualizationPublisher:
    def __init__(self, settings: VisualizationPublisherSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop_requested = False
        self._latest_snapshot: VisualizationSnapshot | None = None
        self._last_error_message: str | None = None
        self._worker = threading.Thread(target=self._run, name="visualization-publisher", daemon=True)
        self._worker.start()

    def publish(self, snapshot: VisualizationSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot
        self._event.set()

    def close(self) -> None:
        self._stop_requested = True
        self._event.set()
        self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()

            if self._stop_requested:
                return

            snapshot = self._take_latest_snapshot()
            if snapshot is None:
                continue

            try:
                self._publish_snapshot(snapshot)
                self._last_error_message = None
            except Exception as exc:  # noqa: BLE001
                message = f"Visualization publish failed: {exc}"
                if message != self._last_error_message:
                    print(message)
                    self._last_error_message = message

    def _take_latest_snapshot(self) -> VisualizationSnapshot | None:
        with self._lock:
            snapshot = self._latest_snapshot
            self._latest_snapshot = None
            return snapshot

    def _publish_snapshot(self, snapshot: VisualizationSnapshot) -> None:
        payload = {
            "update_index": snapshot.update_index,
            "image_stamp_seconds": snapshot.image_stamp_seconds,
            "image_stamp_nanoseconds": snapshot.image_stamp_nanoseconds,
            "particles": [
                {
                    "x": particle.x,
                    "y": particle.y,
                    "yaw": particle.yaw,
                    "weight": particle.weight,
                }
                for particle in snapshot.particles
            ],
            "estimated_pose": {
                "x": snapshot.estimated_pose.x,
                "y": snapshot.estimated_pose.y,
                "yaw": snapshot.estimated_pose.yaw,
            },
            "ground_truth_pose": None if snapshot.ground_truth_pose is None else {
                "x": snapshot.ground_truth_pose.x,
                "y": snapshot.ground_truth_pose.y,
                "yaw": snapshot.ground_truth_pose.yaw,
            },
            "metrics": {
                "best_particle_index": snapshot.best_particle_index,
                "best_score": snapshot.best_score,
                "effective_particle_count": snapshot.effective_particle_count,
                "render_and_score_milliseconds": snapshot.render_and_score_milliseconds,
                "resampled": snapshot.resampled,
            },
            "filter_state": {
                "particle_count": snapshot.filter_state.particle_count,
                "resample_threshold_ratio": snapshot.filter_state.resample_threshold_ratio,
                "measurement": {
                    "temperature": snapshot.filter_state.temperature,
                },
                "motion_noise": {
                    "x_meters": snapshot.filter_state.motion_noise_x_meters,
                    "y_meters": snapshot.filter_state.motion_noise_y_meters,
                    "yaw_radians": snapshot.filter_state.motion_noise_yaw_radians,
                },
                "runtime": {
                    "paused": snapshot.filter_state.paused,
                },
            },
            "images": {
                "observation_jpeg_base64": self._encode_observation_image(snapshot),
                "best_render_png_base64": base64.b64encode(snapshot.best_render_png_bytes).decode("ascii"),
            },
        }

        request = urllib.request.Request(
            self._settings.publish_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._settings.request_timeout_seconds):
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _encode_observation_image(self, snapshot: VisualizationSnapshot) -> str:
        buffer = io.BytesIO()
        Image.fromarray(snapshot.observation_image_rgb, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=self._settings.observation_jpeg_quality,
            optimize=False,
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_visualization_publisher(settings: VisualizationPublisherSettings) -> VisualizationPublisher:
    if not settings.enabled:
        return NoOpVisualizationPublisher()
    return LatestOnlyHttpVisualizationPublisher(settings)
