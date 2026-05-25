from __future__ import annotations

import argparse
import io
import json
import mimetypes
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from PIL import Image
import yaml

from frontend.reset_command_store import PendingResetCommandStore
from frontend.snapshot_store import LatestSnapshotStore


mimetypes.add_type("text/babel", ".jsx")
mimetypes.add_type("text/javascript", ".js")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "frontend"
MAP_YAML_PATH = PROJECT_ROOT / "map.yaml"
SNAPSHOT_STORE = LatestSnapshotStore()
CONTROL_COMMAND_STORE = PendingResetCommandStore()
CONFIG_PATH = PROJECT_ROOT / "turtlebot_localization.yaml"


def load_map_metadata() -> dict | None:
    if not MAP_YAML_PATH.is_file():
        return None

    with MAP_YAML_PATH.open("r", encoding="utf-8") as map_file:
        raw_metadata = yaml.safe_load(map_file) or {}

    image_name = raw_metadata.get("image")
    if not image_name:
        return None

    image_path = (MAP_YAML_PATH.parent / image_name).resolve()
    if not image_path.is_file():
        return None

    with Image.open(image_path) as image:
        width, height = image.size

    origin = raw_metadata.get("origin", [-10.0, -10.0, 0.0])
    return {
        "image_path": image_path,
        "image_name": image_path.name,
        "width": width,
        "height": height,
        "resolution": float(raw_metadata.get("resolution", 0.05)),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "negate": int(raw_metadata.get("negate", 0)),
        "occupied_thresh": float(raw_metadata.get("occupied_thresh", 0.65)),
        "free_thresh": float(raw_metadata.get("free_thresh", 0.196)),
    }


MAP_METADATA = load_map_metadata()


def load_reset_defaults() -> dict:
    defaults = {"sigma_x": 0.5, "sigma_y": 0.5, "sigma_yaw": 0.5, "yaw": 0.0}
    if not CONFIG_PATH.is_file():
        return defaults
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    prior = raw_config.get("initial_pose_prior", {})
    mean = prior.get("mean", {})
    defaults["yaw"] = float(mean.get("yaw", defaults["yaw"]))
    defaults["sigma_x"] = float(prior.get("sigma_x", defaults["sigma_x"]))
    defaults["sigma_y"] = float(prior.get("sigma_y", defaults["sigma_y"]))
    defaults["sigma_yaw"] = float(prior.get("sigma_yaw", defaults["sigma_yaw"]))
    return defaults


RESET_DEFAULTS = load_reset_defaults()


def load_filter_config() -> dict:
    defaults = {
        "particle_count": 256,
        "resample_threshold_ratio": 0.5,
        "initial_pose_prior": {
            "mean": {"x": -2.685, "y": -2.003, "yaw": -0.020},
            "sigma_x": 0.5,
            "sigma_y": 0.5,
            "sigma_yaw": 0.5,
        },
        "measurement": {"metric_name": "hybrid", "temperature": 0.02, "packed": False, "radius_clip": 3.0},
        "motion_noise": {"x_meters": 0.02, "y_meters": 0.02, "yaw_radians": 0.017453292519943295},
        "runtime": {"paused": False},
        "initialization": {"mode": "local"},
    }
    if not CONFIG_PATH.is_file():
        return defaults
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    initial_prior = raw_config.get("initial_pose_prior", {})
    initial_mean = initial_prior.get("mean", {})
    measurement = raw_config.get("measurement", {})
    motion_noise = raw_config.get("motion_noise", {})
    particle_filter = raw_config.get("particle_filter", {})
    initialization = raw_config.get("initialization", {})
    return {
        "particle_count": int(particle_filter.get("particle_count", defaults["particle_count"])),
        "resample_threshold_ratio": float(particle_filter.get("resample_threshold_ratio", defaults["resample_threshold_ratio"])),
        "initial_pose_prior": {
            "mean": {
                "x": float(initial_mean.get("x", defaults["initial_pose_prior"]["mean"]["x"])),
                "y": float(initial_mean.get("y", defaults["initial_pose_prior"]["mean"]["y"])),
                "yaw": float(initial_mean.get("yaw", defaults["initial_pose_prior"]["mean"]["yaw"])),
            },
            "sigma_x": float(initial_prior.get("sigma_x", defaults["initial_pose_prior"]["sigma_x"])),
            "sigma_y": float(initial_prior.get("sigma_y", defaults["initial_pose_prior"]["sigma_y"])),
            "sigma_yaw": float(initial_prior.get("sigma_yaw", defaults["initial_pose_prior"]["sigma_yaw"])),
        },
        "measurement": {
            "metric_name": measurement.get("metric_name", defaults["measurement"]["metric_name"]),
            "temperature": float(measurement.get("temperature", defaults["measurement"]["temperature"])),
            "packed": bool(measurement.get("packed", defaults["measurement"]["packed"])),
            "radius_clip": float(measurement.get("radius_clip", defaults["measurement"]["radius_clip"])),
        },
        "motion_noise": {
            "x_meters": float(motion_noise.get("x_meters", defaults["motion_noise"]["x_meters"])),
            "y_meters": float(motion_noise.get("y_meters", defaults["motion_noise"]["y_meters"])),
            "yaw_radians": float(motion_noise.get("yaw_radians", defaults["motion_noise"]["yaw_radians"])),
        },
        "runtime": {"paused": False},
        "initialization": {
            "mode": str(initialization.get("mode", defaults["initialization"]["mode"])).strip().lower(),
        },
    }


BASE_FILTER_CONFIG = load_filter_config()


def capabilities_payload() -> dict:
    return {
        "set_prior": True,
        "global_reset": True,
        "prior_preset": True,
        "particle_count": True,
        "resample_threshold": True,
        "temperature": True,
        "motion_noise": True,
        "pause_resume": True,
        "single_step": True,
        "localization_mode": True,
    }


def current_filter_config() -> dict:
    config = deepcopy(BASE_FILTER_CONFIG)
    snapshot = SNAPSHOT_STORE.read()
    if snapshot and isinstance(snapshot, dict):
        filter_state = snapshot.get("filter_state") or {}
        if filter_state:
            if "particle_count" in filter_state:
                config["particle_count"] = int(filter_state["particle_count"])
            if "resample_threshold_ratio" in filter_state:
                config["resample_threshold_ratio"] = float(filter_state["resample_threshold_ratio"])
            measurement = filter_state.get("measurement") or {}
            if "temperature" in measurement:
                config["measurement"]["temperature"] = float(measurement["temperature"])
            motion_noise = filter_state.get("motion_noise") or {}
            for key in ("x_meters", "y_meters", "yaw_radians"):
                if key in motion_noise:
                    config["motion_noise"][key] = float(motion_noise[key])
            runtime = filter_state.get("runtime") or {}
            if "paused" in runtime:
                config["runtime"]["paused"] = bool(runtime["paused"])
            initialization = filter_state.get("initialization") or {}
            if "mode" in initialization:
                config["initialization"]["mode"] = str(initialization["mode"])
    config["capabilities"] = capabilities_payload()
    return config


class VisualizationRequestHandler(BaseHTTPRequestHandler):
    server_version = "ParticleFilterVisualizationServer/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "snapshot_available": SNAPSHOT_STORE.read() is not None,
                    "map_available": MAP_METADATA is not None,
                    "pending_control": CONTROL_COMMAND_STORE.peek() is not None,
                },
            )
            return

        if self.path == "/api/latest":
            snapshot = SNAPSHOT_STORE.read()
            if snapshot is None:
                self._write_json(HTTPStatus.NO_CONTENT, {"status": "empty"})
                return
            self._write_json(HTTPStatus.OK, snapshot)
            return

        if self.path in {"/api/reset-particle-filter/pending", "/api/control-command/pending"}:
            command = CONTROL_COMMAND_STORE.peek()
            if command is None:
                self._write_json(HTTPStatus.NO_CONTENT, {"status": "empty"})
                return
            self._write_json(HTTPStatus.OK, command)
            return

        if self.path in {"/api/reset-particle-filter/next", "/api/control-command/next"}:
            command = CONTROL_COMMAND_STORE.pop()
            if command is None:
                self._write_empty(HTTPStatus.NO_CONTENT)
                return
            self._write_json(HTTPStatus.OK, command)
            return

        if self.path == "/api/reset-defaults":
            self._write_json(
                HTTPStatus.OK,
                {
                    "yaw": RESET_DEFAULTS["yaw"],
                    "sigma_x": RESET_DEFAULTS["sigma_x"],
                    "sigma_y": RESET_DEFAULTS["sigma_y"],
                    "sigma_yaw": RESET_DEFAULTS["sigma_yaw"],
                },
            )
            return

        if self.path == "/api/filter-config":
            self._write_json(HTTPStatus.OK, current_filter_config())
            return

        if self.path == "/api/map-metadata":
            if MAP_METADATA is None:
                self._write_json(HTTPStatus.NO_CONTENT, {"status": "missing"})
                return
            self._write_json(
                HTTPStatus.OK,
                {
                    "image_url": "/api/map-image",
                    "width": MAP_METADATA["width"],
                    "height": MAP_METADATA["height"],
                    "resolution": MAP_METADATA["resolution"],
                    "origin": MAP_METADATA["origin"],
                    "negate": MAP_METADATA["negate"],
                    "occupied_thresh": MAP_METADATA["occupied_thresh"],
                    "free_thresh": MAP_METADATA["free_thresh"],
                },
            )
            return

        if self.path == "/api/map-image":
            if MAP_METADATA is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Map image not configured"})
                return
            self._write_map_image()
            return

        self._serve_static_file()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/publish-latest":
            payload = self._read_json_payload()
            if payload is None:
                return
            SNAPSHOT_STORE.update(payload)
            self._write_json(HTTPStatus.OK, {"status": "stored"})
            return

        if self.path == "/api/reset-particle-filter":
            payload = self._read_json_payload()
            if payload is None:
                return
            try:
                prior_payload = payload["prior"]
                mean_payload = prior_payload["mean"]
                command = {
                    "type": "reset_particle_filter",
                    "issued_at_unix_seconds": time.time(),
                    "prior": {
                        "mean": {
                            "x": float(mean_payload["x"]),
                            "y": float(mean_payload["y"]),
                            "yaw": float(mean_payload["yaw"]),
                        },
                        "sigma_x": float(prior_payload["sigma_x"]),
                        "sigma_y": float(prior_payload["sigma_y"]),
                        "sigma_yaw": float(prior_payload["sigma_yaw"]),
                    },
                }
            except (KeyError, TypeError, ValueError) as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"detail": f"Invalid reset payload: {exc}"})
                return

            CONTROL_COMMAND_STORE.set(command)
            self._write_json(HTTPStatus.OK, {"status": "queued", "command": command})
            return

        if self.path == "/api/control-command":
            payload = self._read_json_payload()
            if payload is None:
                return
            command, error = self._normalize_control_command(payload)
            if error is not None:
                self._write_json(HTTPStatus.BAD_REQUEST, {"detail": error})
                return
            CONTROL_COMMAND_STORE.set(command)
            self._write_json(HTTPStatus.OK, {"status": "queued", "command": command})
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _normalize_control_command(self, payload: dict) -> tuple[dict | None, str | None]:
        command_type = payload.get("type")
        if not isinstance(command_type, str):
            return None, "Missing control command type"

        issued_at = time.time()
        if command_type in {
            "pause_particle_filter",
            "resume_particle_filter",
            "step_particle_filter",
            "global_reset_particle_filter",
        }:
            return {"type": command_type, "issued_at_unix_seconds": issued_at}, None

        if command_type == "set_localization_mode":
            mode = str(payload.get("mode", "")).strip().lower()
            if mode not in {"local", "global"}:
                return None, f"Unsupported localization mode: {mode!r}"
            return {
                "type": command_type,
                "issued_at_unix_seconds": issued_at,
                "mode": mode,
            }, None

        if command_type == "set_particle_filter_parameters":
            command: dict = {"type": command_type, "issued_at_unix_seconds": issued_at}
            if "particle_count" in payload:
                command["particle_count"] = int(payload["particle_count"])
            if "resample_threshold_ratio" in payload:
                command["resample_threshold_ratio"] = float(payload["resample_threshold_ratio"])
            if "temperature" in payload:
                command["temperature"] = float(payload["temperature"])
            if "motion_noise" in payload:
                motion_noise = payload["motion_noise"]
                command["motion_noise"] = {
                    "x_meters": float(motion_noise.get("x_meters")),
                    "y_meters": float(motion_noise.get("y_meters")),
                    "yaw_radians": float(motion_noise.get("yaw_radians")),
                }
            if len(command) <= 2:
                return None, "No parameter fields provided"
            return command, None

        return None, f"Unsupported control command type: {command_type}"

    def _read_json_payload(self) -> dict | None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"detail": f"Invalid JSON: {exc}"})
            return None

    def _serve_static_file(self) -> None:
        request_path = "/index.html" if self.path in {"/", ""} else self.path
        normalized_path = Path(unquote(request_path.lstrip("/")))
        target_path = (WEB_ROOT / normalized_path).resolve()
        allowed_suffixes = {".html", ".js", ".jsx", ".css", ".png", ".jpg", ".jpeg", ".svg", ".ico"}

        if (
            not str(target_path).startswith(str(WEB_ROOT.resolve()))
            or not target_path.is_file()
            or target_path.suffix not in allowed_suffixes
        ):
            self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
            return

        mime_type, _ = mimetypes.guess_type(target_path.name)
        content = target_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_map_image(self) -> None:
        with Image.open(MAP_METADATA["image_path"]) as image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = buffer.getvalue()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _write_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        response_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the particle filter visualization frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), VisualizationRequestHandler)
    print(f"Visualization frontend listening at http://{args.host}:{args.port}")
    if MAP_METADATA is not None:
        print(
            f"Using map underlay: {MAP_METADATA['image_name']} | "
            f"resolution={MAP_METADATA['resolution']:.3f} | origin={MAP_METADATA['origin']}"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
