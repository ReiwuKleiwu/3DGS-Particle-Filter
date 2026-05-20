from __future__ import annotations

import threading
import time
from copy import deepcopy


class LatestSnapshotStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._received_at_unix_seconds: float | None = None

    def update(self, snapshot: dict) -> None:
        with self._lock:
            self._snapshot = deepcopy(snapshot)
            self._received_at_unix_seconds = time.time()

    def read(self) -> dict | None:
        with self._lock:
            if self._snapshot is None:
                return None
            snapshot = deepcopy(self._snapshot)
            snapshot["received_at_unix_seconds"] = self._received_at_unix_seconds
            return snapshot
