from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 640
    depth_height: int = 480
    fps: int = 30
    depth_window_radius: int = 2
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50


@dataclass(frozen=True)
class RobotConfig:
    host: str = "127.0.0.1"
    port: int = 18765
    allow_motion: bool = False
    default_speed: int = 5
    motion_timeout_s: float = 120.0
    service_script: str = (
        "C:/ai/robot_arm/robot_arm_workflow_ui/robot_arm_workflow_ui/robot_service.py"
    )


@dataclass(frozen=True)
class GripperConfig:
    port: str = "COM11"
    baud_rate: int = 115200
    modbus_id: int = 1
    stop_bits: int = 1
    parity: str = "N"


@dataclass(frozen=True)
class SensorSideConfig:
    port: str
    baud_rate: int = 1_000_000


@dataclass(frozen=True)
class SensorsConfig:
    left: SensorSideConfig
    right: SensorSideConfig
    horizontal_flip: bool = True
    stale_after_s: float = 1.0
    median_window: int = 3


@dataclass(frozen=True)
class CalibrationConfig:
    output_path: str = "config/calibration.json"
    ransac_threshold_mm: float = 15.0
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class TargetingConfig:
    marker_to_tip_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 400.0)
    maximum_speed: int = 100


@dataclass(frozen=True)
class TransferConfig:
    grasp_clearance_mm: float = 0.0
    pick_lift_mm: float = 100.0
    place_approach_mm: float = 100.0
    place_clearance_mm: float = 30.0
    transit_speed: int = 5
    approach_speed: int = 3
    acceleration: int = 10
    preview_ttl_s: float = 120.0
    gripper_open_position: int = 1000
    gripper_min_position: int = 100
    gripper_close_step: int = 25
    gripper_speed: int = 20
    gripper_force: int = 50
    grasp_poll_interval_s: float = 0.10
    release_wait_s: float = 0.50
    arrival_position_tolerance_mm: float = 3.0
    arrival_orientation_tolerance_deg: float = 2.0
    arrival_stable_frames: int = 3
    arrival_timeout_s: float = 10.0
    arrival_poll_interval_s: float = 0.10


@dataclass(frozen=True)
class VisionConfig:
    roi_xyxy: tuple[int, int, int, int] = (0, 0, 640, 480)
    green_hsv_low: tuple[int, int, int] = (35, 55, 35)
    green_hsv_high: tuple[int, int, int] = (90, 255, 255)
    gray_hsv_low: tuple[int, int, int] = (0, 0, 45)
    gray_hsv_high: tuple[int, int, int] = (179, 65, 220)
    object_min_area_px: float = 500.0
    object_max_area_px: float = 12_000.0
    tray_min_area_px: float = 1_200.0
    tray_min_saturation: float = 120.0
    tray_min_rotated_aspect_ratio: float = 1.15
    tray_max_rotated_aspect_ratio: float = 1.85
    tray_max_vertices: int = 5
    tray_depth_max_spread_m: float = 0.10
    contour_min_solidity: float = 0.78
    cylinder_min_circularity: float = 0.58
    square_aspect_tolerance: float = 0.28
    depth_max_spread_m: float = 0.08
    morphology_kernel_px: int = 5
    proposal_min_height_m: float = 0.012
    proposal_max_height_m: float = 0.35
    proposal_min_area_px: float = 350.0
    proposal_max_area_px: float = 80_000.0
    gray_depth_min_fraction: float = 0.55
    depth_square_aspect_tolerance: float = 0.65
    gray_adaptive_block_px: int = 51
    gray_adaptive_c: float = 10.0


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    robot: RobotConfig
    gripper: GripperConfig
    sensors: SensorsConfig
    calibration: CalibrationConfig
    targeting: TargetingConfig
    transfer: TransferConfig
    vision: VisionConfig


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.json"


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else default_config_path()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    sensors_raw = raw.get("sensors", {})
    return AppConfig(
        camera=CameraConfig(**raw.get("camera", {})),
        robot=RobotConfig(**raw.get("robot", {})),
        gripper=GripperConfig(**raw.get("gripper", {})),
        sensors=SensorsConfig(
            left=SensorSideConfig(**sensors_raw.get("left", {"port": "COM78"})),
            right=SensorSideConfig(**sensors_raw.get("right", {"port": "COM79"})),
            horizontal_flip=bool(sensors_raw.get("horizontal_flip", True)),
            stale_after_s=float(sensors_raw.get("stale_after_s", 1.0)),
            median_window=int(sensors_raw.get("median_window", 3)),
        ),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        targeting=TargetingConfig(
            **{
                **raw.get("targeting", {}),
                "marker_to_tip_xyz_mm": tuple(
                    raw.get("targeting", {}).get("marker_to_tip_xyz_mm", [0, 0, 400])
                ),
            }
        ),
        transfer=TransferConfig(**raw.get("transfer", {})),
        vision=VisionConfig(
            **{
                **raw.get("vision", {}),
                "roi_xyxy": tuple(raw.get("vision", {}).get("roi_xyxy", [0, 0, 640, 480])),
                "green_hsv_low": tuple(raw.get("vision", {}).get("green_hsv_low", [35, 55, 35])),
                "green_hsv_high": tuple(
                    raw.get("vision", {}).get("green_hsv_high", [90, 255, 255])
                ),
                "gray_hsv_low": tuple(raw.get("vision", {}).get("gray_hsv_low", [0, 0, 45])),
                "gray_hsv_high": tuple(raw.get("vision", {}).get("gray_hsv_high", [179, 65, 220])),
            }
        ),
    )
