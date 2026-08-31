from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from .camera_app import ClickResult, RealSenseClickApp
from .config import AppConfig, load_config
from .robot_client import RM65Client

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return Path("records") / "calibration" / f"blue-dot-{stamp}.jsonl"


class CalibrationRecorder:
    """Append-only JSONL writer for auditable calibration observations."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0
        self.last_sample: dict[str, Any] | None = None
        self._next_sample_index = 1
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"refusing to overwrite calibration file: {self.path}")
        self._append(
            {
                "type": "session",
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "marker": "blue_dot",
                **metadata,
            }
        )

    @classmethod
    def resume(cls, path: str | Path) -> CalibrationRecorder:
        """Open a valid calibration JSONL record and append new samples to it."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"calibration file does not exist: {source}")

        session_found = False
        sample_count = 0
        max_sample_index = 0
        last_sample: dict[str, Any] | None = None
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid calibration JSON") from exc
            if not isinstance(event, dict):
                raise TypeError(f"line {line_number}: calibration event must be an object")
            if event.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"line {line_number}: unsupported calibration schema")
            if event.get("type") == "session":
                if session_found or sample_count:
                    raise ValueError(f"line {line_number}: unexpected calibration session event")
                session_found = True
            elif event.get("type") == "sample":
                if not session_found:
                    raise ValueError(f"line {line_number}: sample appears before session event")
                try:
                    sample_index = int(event["sample_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"line {line_number}: invalid sample_index") from exc
                if sample_index <= max_sample_index:
                    raise ValueError(f"line {line_number}: sample_index is not increasing")
                sample_count += 1
                max_sample_index = sample_index
                last_sample = event
            else:
                raise ValueError(f"line {line_number}: unknown calibration event type")

        if not session_found:
            raise ValueError("calibration file does not contain a session event")

        recorder = cls.__new__(cls)
        recorder.path = source
        recorder.sample_count = sample_count
        recorder.last_sample = last_sample
        recorder._next_sample_index = max_sample_index + 1
        return recorder

    def append_sample(
        self,
        click: ClickResult,
        robot_state: dict[str, Any],
    ) -> dict[str, Any]:
        pose = robot_state.get("pose")
        joints = robot_state.get("joints")
        if not isinstance(pose, list) or len(pose) != 6:
            raise ValueError("RM65 response does not contain a six-value TCP pose")
        if not isinstance(joints, list) or not joints:
            raise ValueError("RM65 response does not contain joint values")
        numeric_values = [*pose, *joints, click.depth_m, *click.camera_xyz_m]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("calibration sample contains a non-finite value")

        sample = {
            "type": "sample",
            "schema_version": SCHEMA_VERSION,
            "sample_index": self._next_sample_index,
            "captured_at": utc_now(),
            "pixel_uv": list(click.pixel),
            "depth_m": click.depth_m,
            "camera_xyz_m": list(click.camera_xyz_m),
            "robot_tcp_pose_mm_deg": [float(value) for value in pose],
            "robot_joint_values": [float(value) for value in joints],
        }
        self._append(sample)
        self.sample_count += 1
        self._next_sample_index += 1
        self.last_sample = sample
        return sample

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def prepare_robot_read_only(client: RM65Client) -> dict[str, Any]:
    """Ensure pose feedback is available without enabling or moving the arm."""
    identity = client.ping()
    status = client.status()
    if not status.get("connected"):
        status = client.connect()
    state = client.get_pose()
    return {"identity": identity, "status": status, "initial_state": state}


class CalibrationCollectorApp:
    window_name = "VTLA calibration - click blue dot, S save, Q/Esc exit"

    def __init__(self, config: AppConfig, output_path: Path) -> None:
        self.config = config
        self.output_path = output_path
        self.camera = RealSenseClickApp(config.camera)
        self.robot = RM65Client(config.robot)
        self.recorder: CalibrationRecorder | None = None
        self.status_message = "Click the blue dot, then press S to save"

    def run(self) -> None:
        self.camera.start()
        try:
            robot_info = prepare_robot_read_only(self.robot)
            metadata = self._session_metadata(robot_info)
            self.recorder = CalibrationRecorder(self.output_path, metadata)
            print(f"calibration_output={self.output_path.resolve()}", flush=True)
            print("controls: left-click=select, S=save, Q/Esc=exit", flush=True)

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.camera._on_mouse)
            while True:
                frames = self.camera.align.process(self.camera.pipeline.wait_for_frames())
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                self.camera.latest_depth_raw = np.asanyarray(depth_frame.get_data()).copy()
                self.camera.depth_intrinsics = (
                    depth_frame.profile.as_video_stream_profile().get_intrinsics()
                )
                display = np.asanyarray(color_frame.get_data()).copy()
                self.camera._draw_result(display)
                self._draw_status(display)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key in {ord("s"), ord("S")}:
                    self._save_current()
        finally:
            self.camera.stop()

    def _save_current(self) -> None:
        click = self.camera.latest_result
        if click is None:
            self.status_message = "No valid blue-dot point selected"
            print(self.status_message, flush=True)
            return
        try:
            robot_state = self.robot.get_pose()
            if self.recorder is None:
                raise RuntimeError("calibration recorder is not ready")
            sample = self.recorder.append_sample(click, robot_state)
            self.status_message = f"Saved sample {sample['sample_index']}"
            print(
                f"saved sample={sample['sample_index']} "
                f"camera_xyz_m={sample['camera_xyz_m']} "
                f"tcp_mm_deg={sample['robot_tcp_pose_mm_deg']}",
                flush=True,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            self.status_message = f"Save failed: {exc}"
            print(self.status_message, flush=True)

    def _session_metadata(self, robot_info: dict[str, Any]) -> dict[str, Any]:
        profile = self.camera.pipeline.get_active_profile()
        device = profile.get_device()
        return {
            "camera": {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            },
            "robot": robot_info["identity"],
            "robot_status_at_start": robot_info["status"],
            "units": {
                "camera_xyz": "m",
                "depth": "m",
                "robot_tcp_position": "mm",
                "robot_tcp_orientation": "degree",
                "robot_joint_values": "RM SDK native",
            },
            "notes": (
                "Raw observations only. Blue-dot-to-TCP offset has not been solved; "
                "do not treat these samples as a final camera-to-base transform."
            ),
        }

    def _draw_status(self, image: np.ndarray) -> None:
        count = self.recorder.sample_count if self.recorder else 0
        cv2.putText(
            image,
            f"samples={count} | {self.status_message}",
            (15, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect D435 blue-dot / RM65 pose pairs")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or default_output_path()
    try:
        CalibrationCollectorApp(load_config(args.config), output).run()
    except Exception as exc:
        raise SystemExit(f"calibration error: {exc}") from exc


if __name__ == "__main__":
    main()
