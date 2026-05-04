from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy


class PendingResetCommandStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_commands: deque[dict] = deque()

    def set(self, command: dict) -> None:
        with self._lock:
            self._pending_commands.append(deepcopy(command))

    def pop(self) -> dict | None:
        with self._lock:
            if not self._pending_commands:
                return None
            return deepcopy(self._pending_commands.popleft())

    def peek(self) -> dict | None:
        with self._lock:
            if not self._pending_commands:
                return None
            return deepcopy(self._pending_commands[0])
