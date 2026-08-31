from __future__ import annotations

import threading
from collections import deque
from datetime import UTC, datetime
from typing import Any


class LogStore:
    def __init__(self, capacity: int = 500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._next_id = 1

    def add(self, level: str, source: str, message: str, **details: Any) -> dict[str, Any]:
        with self._lock:
            event = {
                "id": self._next_id,
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "level": level,
                "source": source,
                "message": message,
                "details": details,
            }
            self._next_id += 1
            self._events.append(event)
            return event

    def list(self, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events if event["id"] > after]
