from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlCommandClientSettings:
    enabled: bool = False
    poll_url: str = "http://127.0.0.1:8090/api/reset-particle-filter/next"
    request_timeout_seconds: float = 0.05


class ControlCommandClient:
    def __init__(self, settings: ControlCommandClientSettings) -> None:
        self._settings = settings
        self._last_error_message: str | None = None

    def poll_next_command(self) -> dict | None:
        if not self._settings.enabled:
            return None

        request = urllib.request.Request(
            self._settings.poll_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._settings.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 204:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            self._log_error_once(f"Control command poll failed: HTTP {exc.code}: {detail}")
            return None
        except Exception as exc:  # noqa: BLE001
            self._log_error_once(f"Control command poll failed: {exc}")
            return None

        self._last_error_message = None
        return payload

    def _log_error_once(self, message: str) -> None:
        if message != self._last_error_message:
            print(message)
            self._last_error_message = message
