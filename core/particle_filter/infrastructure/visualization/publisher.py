"""Publishes the latest visualization snapshot to the frontend service."""

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

from core.particle_filter.infrastructure.visualization.models import VisualizationSnapshot


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
                    "recovery_sample": particle.recovery_sample,
                    "roughening_sample": particle.roughening_sample,
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
            "amcl_pose": None if snapshot.amcl_pose is None else {
                "x": snapshot.amcl_pose.x,
                "y": snapshot.amcl_pose.y,
                "yaw": snapshot.amcl_pose.yaw,
            },
            "metrics": {
                "best_particle_index": snapshot.best_particle_index,
                "best_particle_pose": {
                    "x": snapshot.best_particle_pose.x,
                    "y": snapshot.best_particle_pose.y,
                    "yaw": snapshot.best_particle_pose.yaw,
                },
                "best_score": snapshot.best_score,
                "effective_particle_count": snapshot.effective_particle_count,
                "render_and_score_milliseconds": snapshot.render_and_score_milliseconds,
                "resampled": snapshot.resampled,
                "random_particle_ratio": snapshot.random_particle_ratio,
                "random_particle_count": snapshot.random_particle_count,
                "roughening_particle_count": snapshot.roughening_particle_count,
            },
            "filter_state": {
                "particle_count": snapshot.filter_state.particle_count,
                "resample_threshold_ratio": snapshot.filter_state.resample_threshold_ratio,
                "particle_filter": {
                    "roughening_enabled": snapshot.filter_state.roughening_enabled,
                    "roughening_mode": snapshot.filter_state.roughening_mode,
                    "roughening_ratio": snapshot.filter_state.roughening_ratio,
                    "roughening_sigma_x": snapshot.filter_state.roughening_sigma_x,
                    "roughening_sigma_y": snapshot.filter_state.roughening_sigma_y,
                    "roughening_sigma_yaw": snapshot.filter_state.roughening_sigma_yaw,
                },
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
                "recovery": {
                    "enabled": snapshot.filter_state.recovery_enabled,
                    "strategy": snapshot.filter_state.recovery_strategy,
                    "random_particle_floor_ratio": snapshot.filter_state.recovery_random_particle_floor_ratio,
                    "random_particle_max_ratio": snapshot.filter_state.recovery_random_particle_max_ratio,
                    "absolute_score_profiles": snapshot.filter_state.recovery_absolute_score_profiles,
                },
                "adaptive_particle_count": {
                    "enabled": snapshot.filter_state.adaptive_particle_count.enabled,
                    "min_particle_count": snapshot.filter_state.adaptive_particle_count.min_particle_count,
                    "medium_particle_count": snapshot.filter_state.adaptive_particle_count.medium_particle_count,
                    "max_particle_count": snapshot.filter_state.adaptive_particle_count.max_particle_count,
                    "target_particle_count": snapshot.filter_state.adaptive_particle_count.target_particle_count,
                    "stable_required_updates": snapshot.filter_state.adaptive_particle_count.stable_required_updates,
                    "unstable_required_updates": snapshot.filter_state.adaptive_particle_count.unstable_required_updates,
                    "xy_spread_stable_meters": snapshot.filter_state.adaptive_particle_count.xy_spread_stable_meters,
                    "xy_spread_unstable_meters": snapshot.filter_state.adaptive_particle_count.xy_spread_unstable_meters,
                    "yaw_spread_stable_radians": snapshot.filter_state.adaptive_particle_count.yaw_spread_stable_radians,
                    "yaw_spread_unstable_radians": snapshot.filter_state.adaptive_particle_count.yaw_spread_unstable_radians,
                    "best_score_stable_threshold": snapshot.filter_state.adaptive_particle_count.best_score_stable_threshold,
                    "median_score_stable_threshold": snapshot.filter_state.adaptive_particle_count.median_score_stable_threshold,
                    "stable_update_count": snapshot.filter_state.adaptive_particle_count.stable_update_count,
                    "unstable_update_count": snapshot.filter_state.adaptive_particle_count.unstable_update_count,
                    "xy_spread_meters": snapshot.filter_state.adaptive_particle_count.xy_spread_meters,
                    "yaw_spread_radians": snapshot.filter_state.adaptive_particle_count.yaw_spread_radians,
                    "last_resize_reason": snapshot.filter_state.adaptive_particle_count.last_resize_reason,
                },
                "initialization": {
                    "mode": snapshot.filter_state.localization_mode,
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
