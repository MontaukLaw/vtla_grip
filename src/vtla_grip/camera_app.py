from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from .config import CameraConfig, load_config
from .depth import median_depth_m


@dataclass
class ClickResult:
    pixel: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]


class RealSenseClickApp:
    window_name = "VTLA D435 - click target, Q/Esc to exit"

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.depth_intrinsics = None
        self.latest_depth_raw: np.ndarray | None = None
        self.latest_result: ClickResult | None = None

    def start(self) -> None:
        context = rs.context()
        devices = context.query_devices()
        if not devices:
            raise RuntimeError("no RealSense device found")
        device = devices[0]
        if device.supports(rs.camera_info.usb_type_descriptor):
            usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
            try:
                usb_major = int(usb_type.split(".", maxsplit=1)[0])
            except (ValueError, IndexError):
                usb_major = 0
            if usb_major < 3:
                raise RuntimeError(
                    f"D435 is connected as USB {usb_type}; use a USB 3.x port and cable "
                    "for aligned color and depth streaming"
                )
        pipeline_config = rs.config()
        pipeline_config.enable_stream(
            rs.stream.depth,
            self.config.depth_width,
            self.config.depth_height,
            rs.format.z16,
            self.config.fps,
        )
        pipeline_config.enable_stream(
            rs.stream.color,
            self.config.color_width,
            self.config.color_height,
            rs.format.bgr8,
            self.config.fps,
        )
        profile = self.pipeline.start(pipeline_config)
        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())

    def stop(self) -> None:
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def _on_mouse(self, event: int, u: int, v: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.latest_depth_raw is None or self.depth_intrinsics is None:
            return
        depth_m = median_depth_m(
            self.latest_depth_raw,
            u,
            v,
            self.depth_scale,
            radius=self.config.depth_window_radius,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        if depth_m is None:
            self.latest_result = None
            print(f"pixel=({u}, {v}) depth=invalid", flush=True)
            return
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth_m)
        xyz = tuple(float(value) for value in point)
        self.latest_result = ClickResult((u, v), depth_m, xyz)
        print(
            f"pixel=({u}, {v}) depth={depth_m:.4f} m "
            f"camera_xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) m",
            flush=True,
        )

    def run(self) -> None:
        self.start()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                frames = self.align.process(self.pipeline.wait_for_frames())
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                self.latest_depth_raw = np.asanyarray(depth_frame.get_data()).copy()
                self.depth_intrinsics = (
                    depth_frame.profile.as_video_stream_profile().get_intrinsics()
                )
                display = np.asanyarray(color_frame.get_data()).copy()
                self._draw_result(display)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
        finally:
            self.stop()

    def check(self, frame_count: int = 10) -> None:
        """Acquire aligned frames without opening a GUI and print a hardware summary."""
        self.start()
        try:
            aligned_depth = None
            color_frame = None
            for _ in range(frame_count):
                frames = self.align.process(self.pipeline.wait_for_frames(5000))
                aligned_depth = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
            if not aligned_depth or not color_frame:
                raise RuntimeError("D435 did not return aligned color and depth frames")
            depth_raw = np.asanyarray(aligned_depth.get_data())
            height, width = depth_raw.shape
            centre_depth = median_depth_m(
                depth_raw,
                width // 2,
                height // 2,
                self.depth_scale,
                radius=self.config.depth_window_radius,
                min_depth_m=self.config.min_depth_m,
                max_depth_m=self.config.max_depth_m,
            )
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            firmware = device.get_info(rs.camera_info.firmware_version)
            color_shape = np.asanyarray(color_frame.get_data()).shape
            print(f"device={name}")
            print(f"serial={serial} firmware={firmware}")
            print(f"color_shape={color_shape} aligned_depth_shape={depth_raw.shape}")
            print(f"depth_scale={self.depth_scale:g} centre_depth_m={centre_depth}")
        finally:
            self.stop()

    def _draw_result(self, image: np.ndarray) -> None:
        result = self.latest_result
        if result is None:
            cv2.putText(
                image,
                "Click an object to read camera XYZ",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            return
        u, v = result.pixel
        x, y, z = result.camera_xyz_m
        cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        text = f"uv=({u},{v}) XYZ=({x:.3f},{y:.3f},{z:.3f})m"
        cv2.putText(
            image,
            text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="D435 aligned depth click-to-XYZ viewer")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="acquire ten aligned frames and exit without opening a window",
    )
    args = parser.parse_args()
    app = RealSenseClickApp(load_config(args.config).camera)
    try:
        if args.check:
            app.check()
        else:
            app.run()
    except RuntimeError as exc:
        raise SystemExit(f"camera error: {exc}") from exc


if __name__ == "__main__":
    main()
