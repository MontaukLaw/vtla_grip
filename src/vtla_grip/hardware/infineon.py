from __future__ import annotations

import struct
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np
import serial

from ..config import SensorsConfig, SensorSideConfig
from ..sensor_settings import SensorSettingsStore
from .signal_processing import GraspSuccessJudge, SensorSignalProcessor

CHANNEL_COUNT = 32
DATA_MAGIC = b"DATA"
ALGO_MAGIC = b"ALGO"
STATE_MAGIC = b"STAT"
DATA_HEADER_SIZE = 10
DATA_PROTOCOL_VERSION = 1
DATA_FRAME_SIGNAL = 1
DATA_FRAME_DEBUG = 2


class SensorProtocolError(RuntimeError):
    pass


class InfineonFrameParser:
    """Incremental parser matching arm_gripping's 32-channel Infineon formats."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.invalid_headers = 0

    def feed(self, data: bytes) -> list[list[float]]:
        self.buffer.extend(data)
        frames: list[list[float]] = []
        while True:
            data_index = self._find_data_header()
            if data_index is not None:
                if data_index > 0:
                    del self.buffer[:data_index]
                    continue
                parsed = self._parse_data_frame()
                if parsed is None:
                    break
                frames.extend(parsed)
                continue
            legacy = self._parse_legacy_frame()
            if legacy is not None:
                if legacy:
                    frames.append(legacy)
                continue
            keep = 260 + len(ALGO_MAGIC) - 1
            if len(self.buffer) <= keep:
                break
            del self.buffer[:-keep]
        return frames

    def _find_data_header(self) -> int | None:
        start = 0
        while True:
            index = self.buffer.find(DATA_MAGIC, start)
            if index < 0:
                return None
            if len(self.buffer) < index + DATA_HEADER_SIZE:
                return index if index == 0 else None
            if self._valid_header(index):
                return index
            self.invalid_headers += 1
            start = index + 1

    def _valid_header(self, offset: int = 0) -> bool:
        version = self.buffer[offset + 4]
        frame_type = self.buffer[offset + 5]
        channels = self.buffer[offset + 6]
        payload_length = struct.unpack_from("<H", self.buffer, offset + 8)[0]
        return bool(
            version == DATA_PROTOCOL_VERSION
            and channels == CHANNEL_COUNT
            and (
                (frame_type == DATA_FRAME_SIGNAL and payload_length == CHANNEL_COUNT * 2)
                or (frame_type == DATA_FRAME_DEBUG and payload_length == CHANNEL_COUNT * 5 * 2)
            )
        )

    def _parse_data_frame(self) -> list[list[float]] | None:
        if len(self.buffer) < DATA_HEADER_SIZE:
            return None
        payload_length = struct.unpack_from("<H", self.buffer, 8)[0]
        frame_size = DATA_HEADER_SIZE + payload_length
        if len(self.buffer) < frame_size:
            return None
        payload = self.buffer[DATA_HEADER_SIZE:frame_size]
        del self.buffer[:frame_size]
        values = struct.unpack("<" + "H" * (payload_length // 2), payload)
        return [[float(value) for value in values[:CHANNEL_COUNT]]]

    def _parse_legacy_frame(self) -> list[float] | None:
        candidates = []
        for magic, payload_size, kind in ((ALGO_MAGIC, 256, "algo"), (STATE_MAGIC, 64, "state")):
            index = self.buffer.find(magic)
            if index >= 0:
                candidates.append((index, magic, payload_size, kind))
        if not candidates:
            return None
        index, magic, payload_size, kind = min(candidates)
        if index < payload_size:
            del self.buffer[: index + len(magic)]
            return []
        payload = self.buffer[index - payload_size : index]
        del self.buffer[: index + len(magic)]
        if kind != "algo":
            return []
        values = struct.unpack("<" + "H" * (payload_size // 2), payload)
        return [float(value) for value in values[:CHANNEL_COUNT]]


class InfineonSideReader:
    def __init__(self, config: SensorSideConfig, median_window: int = 3) -> None:
        self.config = config
        self.median_window = max(1, int(median_window))
        self._serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._history: deque[list[float]] = deque(maxlen=self.median_window)
        self._frame_times: deque[float] = deque(maxlen=200)
        self._last_frame_at = 0.0
        self._frame_id = 0
        self._last_error: str | None = None
        self._parser = InfineonFrameParser()

    @property
    def connected(self) -> bool:
        return bool(
            self._serial and self._serial.is_open and self._thread and self._thread.is_alive()
        )

    def open(self) -> None:
        if self.connected:
            return
        try:
            self._serial = serial.Serial(self.config.port, self.config.baud_rate, timeout=0.02)
        except (OSError, serial.SerialException) as exc:
            self._last_error = str(exc)
            raise SensorProtocolError(f"failed to open sensor {self.config.port}: {exc}") from exc
        self._stop.clear()
        self._last_error = None
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"sensor-{self.config.port}"
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial:
            self._serial.close()
        self._serial = None
        self._thread = None

    def snapshot(self) -> list[float] | None:
        with self._lock:
            if not self._history:
                return None
            return np.median(np.asarray(self._history, dtype=np.float64), axis=0).tolist()

    def status(self, stale_after_s: float) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            age = now - self._last_frame_at if self._last_frame_at else None
            recent = [value for value in self._frame_times if now - value <= 1.5]
        fps = 0.0
        if len(recent) >= 2 and recent[-1] > recent[0]:
            fps = (len(recent) - 1) / (recent[-1] - recent[0])
        return {
            "connected": self.connected,
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "frame_ready": age is not None and age <= stale_after_s,
            "frame_age_s": age,
            "fps": fps,
            "last_error": self._last_error,
        }

    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def _read_loop(self) -> None:
        assert self._serial is not None
        try:
            while not self._stop.is_set():
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting else 1)
                if not chunk:
                    continue
                for frame in self._parser.feed(chunk):
                    if len(frame) != CHANNEL_COUNT:
                        continue
                    now = time.monotonic()
                    with self._lock:
                        self._history.append(frame)
                        self._frame_times.append(now)
                        self._last_frame_at = now
                        self._frame_id += 1
        except (OSError, serial.SerialException) as exc:
            self._last_error = str(exc)


class DualInfineonSensors:
    def __init__(self, config: SensorsConfig, settings: SensorSettingsStore | None = None) -> None:
        self.config = config
        self.settings = settings or SensorSettingsStore()
        self.left = InfineonSideReader(config.left, 1)
        self.right = InfineonSideReader(config.right, 1)
        self._left_processor = SensorSignalProcessor(self.settings.side("left"))
        self._right_processor = SensorSignalProcessor(self.settings.side("right"))
        self._grasp_judge = GraspSuccessJudge()
        self._last_processed_ids = (-1, -1)
        self._last_payload: dict[str, Any] | None = None
        self._zero = np.zeros(64, dtype=np.float64)
        self._lock = threading.RLock()

    def connect(
        self, left_port: str | None = None, right_port: str | None = None
    ) -> dict[str, Any]:
        if left_port or right_port:
            self.config = replace(
                self.config,
                left=replace(self.config.left, port=left_port or self.config.left.port),
                right=replace(self.config.right, port=right_port or self.config.right.port),
            )
            self.left = InfineonSideReader(self.config.left, 1)
            self.right = InfineonSideReader(self.config.right, 1)
        try:
            self.left.open()
            self.right.open()
        except SensorProtocolError:
            self.disconnect()
            raise
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.right.close()
        self.left.close()
        return self.status()

    def zero(self) -> dict[str, Any]:
        raw = self._raw_frame()
        if raw is None:
            raise SensorProtocolError("both sensor frames must be ready before zeroing")
        with self._lock:
            self._zero = np.asarray(raw, dtype=np.float64)
            self._left_processor.reset()
            self._right_processor.reset()
            self._grasp_judge.reset()
            self._last_processed_ids = (-1, -1)
            self._last_payload = None
        return self.data()

    def clear_zero(self) -> dict[str, Any]:
        with self._lock:
            self._zero[:] = 0.0
            self._left_processor.reset()
            self._right_processor.reset()
            self._grasp_judge.reset()
            self._last_processed_ids = (-1, -1)
            self._last_payload = None
        return self.data()

    def apply_settings(self) -> None:
        with self._lock:
            self._left_processor.configure(self.settings.side("left"))
            self._right_processor.configure(self.settings.side("right"))
            self._grasp_judge.reset()
            self._last_processed_ids = (-1, -1)
            self._last_payload = None

    def status(self) -> dict[str, Any]:
        left = self.left.status(self.config.stale_after_s)
        right = self.right.status(self.config.stale_after_s)
        return {
            "connected": bool(left["connected"] and right["connected"]),
            "frame_ready": bool(left["frame_ready"] and right["frame_ready"]),
            "left": left,
            "right": right,
        }

    def data(self) -> dict[str, Any]:
        raw = self._raw_frame()
        if raw is None:
            return {
                **self.status(),
                "raw": None,
                "processed": None,
                "display": None,
                "features": None,
                "grasp_success": None,
            }
        with self._lock:
            frame_ids = (self.left.frame_id(), self.right.frame_id())
            if frame_ids == self._last_processed_ids and self._last_payload is not None:
                return {**self.status(), **self._last_payload}
            zeroed = np.asarray(raw) - self._zero
            left_values, left_display, left_debug = self._left_processor.process(
                zeroed[:32].tolist()
            )
            right_values, right_display, right_debug = self._right_processor.process(
                zeroed[32:].tolist()
            )
            left = np.asarray(left_values)
            right = np.asarray(right_values)
            processed = np.concatenate((left, right))
            display = left_display + right_display
            grasp_success = self._grasp_judge.update(
                left, right, self.settings.side("left"), self.settings.side("right")
            )
            payload = {
                "raw": raw,
                "processed": processed.tolist(),
                "display": display,
                "features": {
                    "left": {**self._features(left), **left_debug},
                    "right": {**self._features(right), **right_debug},
                },
                "grasp_success": grasp_success,
            }
            self._last_processed_ids = frame_ids
            self._last_payload = payload
        return {**self.status(), **payload}

    def _raw_frame(self) -> list[float] | None:
        left = self.left.snapshot()
        right = self.right.snapshot()
        if left is None or right is None:
            return None
        if self.config.horizontal_flip:
            left = self._horizontal_flip(left)
            right = self._horizontal_flip(right)
        return left + right

    @staticmethod
    def _horizontal_flip(values: list[float]) -> list[float]:
        result: list[float] = []
        for row in range(8):
            result.extend(reversed(values[row * 4 : (row + 1) * 4]))
        return result

    @staticmethod
    def _features(values: np.ndarray) -> dict[str, float | int]:
        return {
            "sum": float(values.sum()),
            "max": float(values.max(initial=0.0)),
            "nonzero": int(np.count_nonzero(values)),
        }
