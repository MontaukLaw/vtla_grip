from __future__ import annotations

import copy
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .config import TargetingConfig, TransferConfig
from .tactile_recognition import TactileObjectRecognizer

SUPPORTED_TRANSFER_TARGETS = ("green_cylinder", "gray_cube")


class TransferError(RuntimeError):
    pass


class TransferStopped(TransferError):
    pass


class NoContactError(TransferError):
    pass


def build_transfer_plan(
    vision_result: dict[str, Any],
    target_class: str,
    targeting: TargetingConfig,
    transfer: TransferConfig,
    calibration_solution: dict[str, Any],
    destination_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_class not in SUPPORTED_TRANSFER_TARGETS:
        raise TransferError(f"暂不支持搬运类别：{target_class}")
    source = vision_result.get("recommended_object")
    destination = destination_override or vision_result.get("destination")
    if not isinstance(source, dict) or source.get("class_name") != target_class:
        raise TransferError("没有找到符合指令且可抓取的目标物体")
    expected_destination = "specified_position" if destination_override else "green_tray"
    destination_label = "指定位置" if destination_override else "绿色托盘"
    if not isinstance(destination, dict) or destination.get("class_name") != expected_destination:
        raise TransferError(f"没有找到可用的{destination_label}")
    source_xyz = _validated_xyz(source.get("base_xyz_mm"), "目标物体")
    destination_xyz = _validated_xyz(destination.get("base_xyz_mm"), destination_label)
    source_height_mm = _validated_height(source.get("height_above_table_mm"), "目标物体桌面高度")
    orientation = calibration_solution.get("fixed_tool_orientation_deg")
    orientation_range = calibration_solution.get("tool_orientation_range_deg")
    if not isinstance(orientation, list) or len(orientation) != 3:
        raise TransferError("外参缺少固定工具姿态，请重新求解标定")
    if (
        not isinstance(orientation_range, list)
        or len(orientation_range) != 3
        or max(map(abs, orientation_range)) > 2.0
    ):
        raise TransferError("标定时工具姿态变化过大，禁止自动搬运")

    marker_offset = np.asarray(targeting.marker_to_tip_xyz_mm, dtype=np.float64)
    grasp_xyz = source_xyz + marker_offset + np.asarray([0.0, 0.0, transfer.grasp_clearance_mm])
    pick_above_xyz = grasp_xyz + np.asarray([0.0, 0.0, transfer.pick_lift_mm])
    place_xyz = (
        destination_xyz
        + marker_offset
        + np.asarray([0.0, 0.0, source_height_mm + transfer.place_clearance_mm])
    )
    place_above_xyz = place_xyz + np.asarray([0.0, 0.0, transfer.place_approach_mm])
    tool_orientation = [float(value) for value in orientation]

    def pose(xyz: np.ndarray) -> list[float]:
        return [*map(float, xyz.tolist()), *tool_orientation]

    created_at = datetime.now(UTC)
    return {
        "plan_id": uuid.uuid4().hex,
        "created_at": created_at.isoformat(timespec="milliseconds"),
        "created_timestamp": created_at.timestamp(),
        "expires_in_s": transfer.preview_ttl_s,
        "command": {
            "action": "place_at_position" if destination_override else "place_into_tray",
            "target_class": target_class,
            "destination_class": expected_destination,
        },
        "source": _detection_summary(source),
        "destination": _detection_summary(destination),
        "poses": {
            "pick_above": pose(pick_above_xyz),
            "pick": pose(grasp_xyz),
            "place_above": pose(place_above_xyz),
            "place": pose(place_xyz),
        },
        "motion": {
            "transit_speed": transfer.transit_speed,
            "approach_speed": transfer.approach_speed,
            "acceleration": transfer.acceleration,
            "grasp_clearance_mm": transfer.grasp_clearance_mm,
            "pick_lift_mm": transfer.pick_lift_mm,
            "place_approach_mm": transfer.place_approach_mm,
            "place_clearance_mm": transfer.place_clearance_mm,
            "source_height_above_table_mm": source_height_mm,
        },
        "steps": [
            "张开夹爪并进行双侧触觉调零",
            "移动到目标物体上方",
            "垂直下降到抓取高度",
            "逐步闭合，检测到双侧接触后停止",
            "保持夹爪并识别物体种类和软硬度",
            "垂直抬升到目标上方",
            f"移动到{destination_label}上方",
            "垂直下降到放置高度",
            "松开物体",
            f"垂直抬升到{destination_label}上方",
        ],
    }


class TransferTaskManager:
    def __init__(
        self,
        robot: Any,
        gripper: Any,
        sensors: Any,
        config: TransferConfig,
        log_callback: Callable[..., Any] | None = None,
        recognizer: TactileObjectRecognizer | None = None,
    ) -> None:
        self.robot = robot
        self.gripper = gripper
        self.sensors = sensors
        self.config = config
        self.log_callback = log_callback
        self.recognizer = recognizer
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._plan: dict[str, Any] | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "current_step": None,
            "completed_steps": 0,
            "error": None,
            "holding_object": False,
            "started_at": None,
            "finished_at": None,
            "recognition_result": None,
        }

    def set_preview(self, plan: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise TransferError("搬运任务正在执行，不能生成新预览")
            self._plan = copy.deepcopy(plan)
            self._state = {
                "status": "preview_ready",
                "current_step": None,
                "completed_steps": 0,
                "error": None,
                "holding_object": False,
                "started_at": None,
                "finished_at": None,
                "recognition_result": None,
            }
            return self.status()

    def update_config(self, config: TransferConfig) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise TransferError("搬运任务执行中，不能修改运动参数")
            self.config = config
            self._plan = None
            self._state = {
                "status": "idle",
                "current_step": None,
                "completed_steps": 0,
                "error": None,
                "holding_object": False,
                "started_at": None,
                "finished_at": None,
                "recognition_result": None,
            }
            return self.status()

    def clear_preview(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise TransferError("搬运任务执行中，不能修改指定放置点")
            self._plan = None
            self._state = {
                "status": "idle",
                "current_step": None,
                "completed_steps": 0,
                "error": None,
                "holding_object": False,
                "started_at": None,
                "finished_at": None,
                "recognition_result": None,
            }
            return self.status()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **copy.deepcopy(self._state),
                "running": self.running,
                "plan": copy.deepcopy(self._plan),
            }

    def start(self, plan_id: str) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise TransferError("搬运任务已经在执行")
            if self._plan is None or self._plan.get("plan_id") != plan_id:
                raise TransferError("搬运预览不存在或已被替换，请重新生成预览")
            age = time.time() - float(self._plan["created_timestamp"])
            if age > float(self._plan["expires_in_s"]):
                raise TransferError("搬运预览已过期，请重新识别物体")
            self._stop.clear()
            self._state.update(
                status="queued",
                current_step="等待执行",
                completed_steps=0,
                error=None,
                holding_object=False,
                started_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
                finished_at=None,
                recognition_result=None,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(copy.deepcopy(self._plan),),
                name="structured-transfer",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def request_stop(self) -> dict[str, Any]:
        self._stop.set()
        try:
            self.robot.slow_stop()
        except Exception as exc:  # noqa: BLE001 - slow stop must remain best effort.
            self._log("warning", "搬运停止时 RM65 慢停请求失败", error=str(exc))
        with self._lock:
            if self.running:
                self._state["status"] = "stopping"
                self._state["current_step"] = "正在停止"
            return self.status()

    def _run(self, plan: dict[str, Any]) -> None:
        poses = plan["poses"]
        motion = plan["motion"]
        holding = False
        try:
            self._update(status="running")
            self._step(plan["steps"][0], 0)
            self._open_gripper()
            self._prepare_sensors()

            self._step(plan["steps"][1], 1)
            self._move(poses["pick_above"], "MoveJ_P", motion["transit_speed"], motion)
            self._step(plan["steps"][2], 2)
            self._move(poses["pick"], "MoveL", motion["approach_speed"], motion)
            self._step(plan["steps"][3], 3)
            try:
                grasp_result = self._grasp_until_contact()
            except NoContactError:
                self._open_gripper()
                self._move(poses["pick_above"], "MoveL", motion["approach_speed"], motion)
                raise
            holding = True
            self._update(holding_object=True, grasp_result=grasp_result)

            self._step(plan["steps"][4], 4)
            self._recognize_held_object(plan["command"]["target_class"], grasp_result)
            self._step(plan["steps"][5], 5)
            self._move(poses["pick_above"], "MoveL", motion["approach_speed"], motion)
            self._step(plan["steps"][6], 6)
            self._move(poses["place_above"], "MoveJ_P", motion["transit_speed"], motion)
            self._step(plan["steps"][7], 7)
            self._move(poses["place"], "MoveL", motion["approach_speed"], motion)
            self._step(plan["steps"][8], 8)
            self._open_gripper()
            holding = False
            self._update(holding_object=False)
            self._step(plan["steps"][9], 9)
            self._move(poses["place_above"], "MoveL", motion["approach_speed"], motion)
            self._update(
                status="completed",
                current_step="搬运完成",
                completed_steps=len(plan["steps"]),
                finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            )
            self._log("info", "结构化搬运任务完成", plan_id=plan["plan_id"])
        except TransferStopped as exc:
            self._finish("stopped", str(exc), holding)
        except Exception as exc:  # noqa: BLE001 - background task boundary records all failures.
            self._finish("failed", str(exc), holding)

    def _step(self, label: str, completed_steps: int) -> None:
        self._check_stop()
        self._update(current_step=label, completed_steps=completed_steps)
        self._log("info", label)

    def _move(self, pose: list[float], mode: str, speed: int, motion: dict[str, Any]) -> None:
        self._check_stop()
        self.robot.move(pose, mode, int(speed), int(motion["acceleration"]))
        self._wait_for_arrival(pose)
        self._check_stop()

    def _wait_for_arrival(self, target_pose: list[float]) -> None:
        deadline = time.monotonic() + self.config.arrival_timeout_s
        stable_frames = 0
        last_position_error = float("inf")
        last_orientation_error = float("inf")
        target = np.asarray(target_pose, dtype=np.float64)
        while time.monotonic() < deadline:
            self._check_stop()
            feedback = self.robot.pose()
            actual_raw = feedback.get("pose") if isinstance(feedback, dict) else None
            if isinstance(actual_raw, list) and len(actual_raw) == 6:
                actual = np.asarray(actual_raw, dtype=np.float64)
                if np.all(np.isfinite(actual)):
                    last_position_error = float(np.linalg.norm(actual[:3] - target[:3]))
                    angle_delta = (actual[3:] - target[3:] + 180.0) % 360.0 - 180.0
                    last_orientation_error = float(np.max(np.abs(angle_delta)))
                    if (
                        last_position_error <= self.config.arrival_position_tolerance_mm
                        and last_orientation_error <= self.config.arrival_orientation_tolerance_deg
                    ):
                        stable_frames += 1
                        if stable_frames >= self.config.arrival_stable_frames:
                            return
                    else:
                        stable_frames = 0
            self._wait(self.config.arrival_poll_interval_s)
        raise TransferError(
            "机械臂未在规定时间内到达目标位姿："
            f"位置误差 {last_position_error:.2f} mm，"
            f"姿态误差 {last_orientation_error:.2f}°"
        )

    def _open_gripper(self) -> None:
        self._check_stop()
        self.gripper.set_force(self.config.gripper_force)
        self.gripper.set_speed(self.config.gripper_speed)
        self.gripper.set_position(self.config.gripper_open_position)
        self._wait(self.config.release_wait_s)

    def _prepare_sensors(self) -> None:
        self.sensors.zero()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self._check_stop()
            data = self.sensors.data()
            features = data.get("features") or {}
            if all(
                bool((features.get(side) or {}).get("baseline_ready")) for side in ("left", "right")
            ):
                return
            self._wait(0.03)
        raise TransferError("触觉传感器基线未能在 3 秒内稳定")

    def _grasp_until_contact(self) -> dict[str, Any]:
        start = self.config.gripper_open_position
        stop = self.config.gripper_min_position
        step = max(1, self.config.gripper_close_step)
        for position in range(start - step, stop - 1, -step):
            self._check_stop()
            command_position = max(stop, position)
            self.gripper.set_position(command_position)
            self._wait(self.config.grasp_poll_interval_s)
            data = self.sensors.data()
            grasp = data.get("grasp_success") or {}
            if grasp.get("success"):
                return {**grasp, "position_command": command_position}
            if command_position == stop:
                break
        raise NoContactError("夹爪到达最小位置仍未检测到双侧接触，物体已原位放回")

    def _recognize_held_object(
        self, vision_class: str, grasp_result: dict[str, Any]
    ) -> None:
        if self.recognizer is None:
            self._update(
                recognition_result={"status": "unavailable", "error": "未配置触觉识别器"}
            )
            return
        started = time.monotonic()
        frames: list[list[float]] = []
        try:
            model = self.recognizer.model_status(vision_class)
            if not model.get("available"):
                raise FileNotFoundError(f"视觉类别 {vision_class} 尚未训练触觉性质模型")
            duration = max(0.1, float(model.get("capture_duration_seconds", 1.0)))
            while time.monotonic() - started < duration:
                self._check_stop()
                processed = self.sensors.data().get("processed")
                if isinstance(processed, list) and len(processed) >= 64:
                    frames.append([float(value) for value in processed[:64]])
                self._wait(0.012)
            if not frames:
                raise RuntimeError("触觉传感器没有提供可用于识别的 64 通道数据")
            final_position: int | None = None
            try:
                raw_position = self.gripper.feedback().position
                if isinstance(raw_position, int) and 0 <= raw_position <= 1000:
                    final_position = raw_position
            except Exception as exc:  # noqa: BLE001 - pressure-only fallback is supported.
                self._log("warning", "读取夹爪最终闭合位置失败", error=str(exc))
            if final_position is None:
                commanded = grasp_result.get("position_command")
                if isinstance(commanded, int) and 0 <= commanded <= 1000:
                    final_position = commanded
            result = self.recognizer.predict(vision_class, frames, final_position)
            payload = {
                "status": "completed",
                "vision_class": result.vision_class,
                "property": result.property,
                "confidence": result.confidence,
                "final_gripper_position": final_position,
                "frame_count": len(frames),
                "capture_duration_s": time.monotonic() - started,
            }
            self._update(recognition_result=payload)
            self._log("info", "抓取后触觉识别完成", **payload)
        except TransferStopped:
            raise
        except Exception as exc:  # noqa: BLE001 - recognition must not drop a held object.
            payload = {
                "status": "failed",
                "error": str(exc),
                "frame_count": len(frames),
                "capture_duration_s": time.monotonic() - started,
            }
            self._update(recognition_result=payload)
            self._log("warning", "抓取后触觉识别失败，继续执行搬运", **payload)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise TransferStopped("用户已停止搬运任务")

    def _wait(self, seconds: float) -> None:
        if self._stop.wait(max(0.0, seconds)):
            self._check_stop()

    def _finish(self, status: str, error: str, holding: bool) -> None:
        self._update(
            status=status,
            current_step="任务已停止" if status == "stopped" else "任务失败",
            error=error,
            holding_object=holding,
            finished_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        )
        self._log("warning" if status == "stopped" else "error", error)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _log(self, level: str, message: str, **data: Any) -> None:
        if self.log_callback is not None:
            self.log_callback(level, "transfer", message, **data)


def _validated_xyz(raw: Any, label: str) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != 3:
        raise TransferError(f"{label}缺少有效的 Base XYZ 坐标")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise TransferError(f"{label}的 Base XYZ 坐标无效")
    return values


def _validated_height(raw: Any, label: str) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TransferError(f"{label}无效，请重新识别物体")
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        raise TransferError(f"{label}无效，请重新识别物体")
    return value


def _detection_summary(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(detection.get(key))
        for key in (
            "detection_id",
            "class_name",
            "confidence",
            "bbox_xywh",
            "center_uv",
            "depth_m",
            "base_xyz_mm",
            "height_above_table_mm",
        )
    }
