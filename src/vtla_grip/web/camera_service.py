from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from ..camera_app import ClickResult, RealSenseClickApp
from ..config import CameraConfig
from ..depth import median_depth_m


class CameraService:
    def __init__(self, config: CameraConfig) -> None:
        self.camera = RealSenseClickApp(config)
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._jpeg: bytes | None = None
        self._color_bgr: np.ndarray | None = None
        self._depth_raw: np.ndarray | None = None
        self._intrinsics = None
        self._last_frame_at = 0.0
        self._frame_sequence = 0
        self._frame_captured_at: str | None = None
        self._error: str | None = None

    def start(self) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return self.status()
        self._stop.clear()
        self._error = None
        self.camera.start()
        self._thread = threading.Thread(target=self._capture_loop, name="d435-capture", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.camera.stop()
        except RuntimeError:
            pass
        with self._lock:
            self._jpeg = None
            self._color_bgr = None
            self._depth_raw = None
            self._intrinsics = None
            self._last_frame_at = 0.0
            self._frame_sequence = 0
            self._frame_captured_at = None
        self._thread = None
        return self.status()

    def status(self) -> dict[str, Any]:
        age = time.monotonic() - self._last_frame_at if self._last_frame_at else None
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "frame_ready": self._jpeg is not None,
            "frame_age_s": age,
            "width": self.config.color_width,
            "height": self.config.color_height,
            "error": self._error,
        }

    def deproject(self, u: int, v: int) -> ClickResult:
        with self._lock:
            if self._depth_raw is None or self._intrinsics is None:
                raise RuntimeError("D435 frame is not ready")
            depth_m = median_depth_m(
                self._depth_raw,
                u,
                v,
                self.camera.depth_scale,
                radius=self.config.depth_window_radius,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m,
            )
            if depth_m is None:
                raise ValueError("selected pixel has no valid depth")
            point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [u, v], depth_m)
        return ClickResult((u, v), depth_m, tuple(float(value) for value in point))

    def deproject_depth(self, u: int, v: int, depth_m: float) -> ClickResult:
        with self._lock:
            if self._intrinsics is None:
                raise RuntimeError("D435 frame is not ready")
            point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [u, v], depth_m)
        return ClickResult((u, v), depth_m, tuple(float(value) for value in point))

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, float]:
        with self._lock:
            if self._color_bgr is None or self._depth_raw is None:
                raise RuntimeError("D435 frame is not ready")
            return self._color_bgr.copy(), self._depth_raw.copy(), self.camera.depth_scale

    def capture_context(self) -> dict[str, Any]:
        with self._lock:
            if self._intrinsics is None or self._color_bgr is None or self._depth_raw is None:
                raise RuntimeError("D435 frame is not ready")
            intrinsics = self._intrinsics
            return {
                "captured_at": self._frame_captured_at,
                "frame_sequence": self._frame_sequence,
                "depth_scale": self.camera.depth_scale,
                "color_shape": list(self._color_bgr.shape),
                "depth_shape": list(self._depth_raw.shape),
                "intrinsics": {
                    "width": int(intrinsics.width),
                    "height": int(intrinsics.height),
                    "ppx": float(intrinsics.ppx),
                    "ppy": float(intrinsics.ppy),
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "model": str(intrinsics.model),
                    "coeffs": [float(value) for value in intrinsics.coeffs],
                },
            }

    def mjpeg(self) -> Iterator[bytes]:
        while True:
            with self._lock:
                frame = self._jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1 / 30)

    def metadata(self) -> dict[str, str]:
        profile = self.camera.pipeline.get_active_profile()
        device = profile.get_device()
        return {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "usb": device.get_info(rs.camera_info.usb_type_descriptor),
        }

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self.camera.align.process(self.camera.pipeline.wait_for_frames(5000))
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                with self._lock:
                    self._jpeg = encoded.tobytes()
                    self._color_bgr = color.copy()
                    self._depth_raw = np.asanyarray(depth_frame.get_data()).copy()
                    self._intrinsics = (
                        depth_frame.profile.as_video_stream_profile().get_intrinsics()
                    )
                    self._last_frame_at = time.monotonic()
                    self._frame_sequence += 1
                    self._frame_captured_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        except RuntimeError as exc:
            self._error = str(exc)
