from __future__ import annotations

import time

import pytest

from vtla_grip.config import TargetingConfig, TransferConfig
from vtla_grip.transfer import TransferError, TransferTaskManager, build_transfer_plan


def _vision_result() -> dict:
    return {
        "recommended_object": {
            "detection_id": "object-1",
            "class_name": "gray_cube",
            "confidence": 0.91,
            "bbox_xywh": [100, 120, 30, 30],
            "center_uv": [115, 135],
            "depth_m": 0.9,
            "base_xyz_mm": [10.0, 20.0, 5.0],
            "height_above_table_mm": 40.0,
        },
        "destination": {
            "detection_id": "tray-1",
            "class_name": "green_tray",
            "confidence": 0.95,
            "bbox_xywh": [300, 260, 80, 60],
            "center_uv": [340, 290],
            "depth_m": 0.85,
            "base_xyz_mm": [110.0, 70.0, 0.0],
        },
    }


def _solution() -> dict:
    return {
        "fixed_tool_orientation_deg": [180.0, 0.0, -5.0],
        "tool_orientation_range_deg": [0.1, 0.0, 0.1],
    }


def test_build_transfer_plan_applies_offsets_and_clearances() -> None:
    plan = build_transfer_plan(
        _vision_result(),
        "gray_cube",
        TargetingConfig(marker_to_tip_xyz_mm=(0.0, 30.0, 100.0)),
        TransferConfig(
            grasp_clearance_mm=-10.0,
            pick_lift_mm=80.0,
            place_approach_mm=90.0,
            place_clearance_mm=25.0,
        ),
        _solution(),
    )

    assert plan["poses"]["pick"] == [10.0, 50.0, 95.0, 180.0, 0.0, -5.0]
    assert plan["poses"]["pick_above"] == [10.0, 50.0, 175.0, 180.0, 0.0, -5.0]
    assert plan["poses"]["place"] == [110.0, 100.0, 165.0, 180.0, 0.0, -5.0]
    assert plan["poses"]["place_above"] == [110.0, 100.0, 255.0, 180.0, 0.0, -5.0]
    assert plan["motion"]["source_height_above_table_mm"] == 40.0


def test_build_transfer_plan_rejects_unsupported_target() -> None:
    with pytest.raises(TransferError, match="暂不支持"):
        build_transfer_plan(
            _vision_result(),
            "green_tray",
            TargetingConfig(),
            TransferConfig(),
            _solution(),
        )


def test_build_transfer_plan_uses_mouse_specified_destination() -> None:
    destination = {
        "detection_id": "specified-position",
        "class_name": "specified_position",
        "confidence": 1.0,
        "bbox_xywh": [312, 232, 16, 16],
        "center_uv": [320, 240],
        "depth_m": 0.9,
        "base_xyz_mm": [210.0, 170.0, 10.0],
    }
    plan = build_transfer_plan(
        _vision_result(),
        "gray_cube",
        TargetingConfig(marker_to_tip_xyz_mm=(0.0, 30.0, 100.0)),
        TransferConfig(place_clearance_mm=25.0),
        _solution(),
        destination_override=destination,
    )

    assert plan["command"] == {
        "action": "place_at_position",
        "target_class": "gray_cube",
        "destination_class": "specified_position",
    }
    assert plan["poses"]["place"] == [210.0, 200.0, 175.0, 180.0, 0.0, -5.0]
    assert "移动到指定位置上方" in plan["steps"]


class _Robot:
    def __init__(self) -> None:
        self.moves: list[tuple[str, list[float]]] = []
        self.stopped = False

    def move(self, pose: list[float], mode: str, speed: int, acc: int) -> dict:
        self.moves.append((mode, pose))
        return {"ok": True, "speed": speed, "acc": acc}

    def pose(self) -> dict:
        pose = self.moves[-1][1] if self.moves else [0.0] * 6
        return {"pose": pose}

    def slow_stop(self) -> dict:
        self.stopped = True
        return {"stopped": True}


class _Gripper:
    def __init__(self) -> None:
        self.positions: list[int] = []

    def set_force(self, value: int) -> dict:
        return {"force": value}

    def set_speed(self, value: int) -> dict:
        return {"speed": value}

    def set_position(self, value: int) -> dict:
        self.positions.append(value)
        return {"position": value}

    def feedback(self):
        return type("Feedback", (), {"position": self.positions[-1]})()


class _Sensors:
    def __init__(self, contact_after: int | None = 3) -> None:
        self.reads = 0
        self.contact_after = contact_after

    def zero(self) -> dict:
        return {}

    def data(self) -> dict:
        self.reads += 1
        return {
            "features": {
                "left": {"baseline_ready": True},
                "right": {"baseline_ready": True},
            },
            "grasp_success": {
                "success": self.contact_after is not None and self.reads >= self.contact_after
            },
            "processed": [float(self.reads)] * 64,
        }


class _Recognizer:
    def model_status(self, vision_class):
        return {"available": True, "capture_duration_seconds": 0.001}

    def predict(self, vision_class, frames, final_gripper_position):
        assert frames and final_gripper_position is not None
        return type(
            "Result",
            (),
            {"vision_class": vision_class, "property": "软", "confidence": 0.9},
        )()


def test_transfer_manager_runs_confirmed_plan_in_safe_order() -> None:
    robot = _Robot()
    gripper = _Gripper()
    manager = TransferTaskManager(
        robot,
        gripper,
        _Sensors(),
        TransferConfig(
            release_wait_s=0.001,
            grasp_poll_interval_s=0.001,
            gripper_close_step=100,
        ),
        recognizer=_Recognizer(),
    )
    plan = build_transfer_plan(
        _vision_result(), "gray_cube", TargetingConfig(), manager.config, _solution()
    )
    manager.set_preview(plan)

    manager.start(plan["plan_id"])
    deadline = time.monotonic() + 2.0
    while manager.running and time.monotonic() < deadline:
        time.sleep(0.005)

    status = manager.status()
    assert status["status"] == "completed"
    assert status["holding_object"] is False
    assert [mode for mode, _ in robot.moves] == [
        "MoveJ_P",
        "MoveL",
        "MoveL",
        "MoveJ_P",
        "MoveL",
        "MoveL",
    ]
    assert gripper.positions[0] == 1000
    assert gripper.positions[-1] == 1000
    assert status["recognition_result"]["vision_class"] == "gray_cube"
    assert status["recognition_result"]["property"] == "软"
    assert status["recognition_result"]["confidence"] == 0.9


def test_gripper_does_not_close_until_robot_arrival_is_confirmed() -> None:
    class DelayedRobot(_Robot):
        def __init__(self) -> None:
            super().__init__()
            self.arrived = False

        def pose(self) -> dict:
            if not self.arrived:
                return {"pose": [0.0] * 6}
            return super().pose()

    robot = DelayedRobot()
    gripper = _Gripper()
    manager = TransferTaskManager(
        robot,
        gripper,
        _Sensors(),
        TransferConfig(
            release_wait_s=0.001,
            grasp_poll_interval_s=0.001,
            gripper_close_step=100,
            arrival_stable_frames=1,
            arrival_timeout_s=1.0,
            arrival_poll_interval_s=0.001,
        ),
    )
    plan = build_transfer_plan(
        _vision_result(), "gray_cube", TargetingConfig(), manager.config, _solution()
    )
    manager.set_preview(plan)

    manager.start(plan["plan_id"])
    deadline = time.monotonic() + 0.5
    while not robot.moves and time.monotonic() < deadline:
        time.sleep(0.001)

    assert robot.moves
    assert gripper.positions == [manager.config.gripper_open_position]

    robot.arrived = True
    deadline = time.monotonic() + 1.0
    while manager.running and time.monotonic() < deadline:
        time.sleep(0.005)

    assert manager.status()["status"] == "completed"
    assert any(position < manager.config.gripper_open_position for position in gripper.positions)


def test_no_contact_opens_gripper_and_returns_above_pick() -> None:
    robot = _Robot()
    gripper = _Gripper()
    manager = TransferTaskManager(
        robot,
        gripper,
        _Sensors(contact_after=None),
        TransferConfig(
            release_wait_s=0.001,
            grasp_poll_interval_s=0.001,
            gripper_min_position=900,
            gripper_close_step=100,
        ),
    )
    plan = build_transfer_plan(
        _vision_result(), "gray_cube", TargetingConfig(), manager.config, _solution()
    )
    manager.set_preview(plan)

    manager.start(plan["plan_id"])
    deadline = time.monotonic() + 2.0
    while manager.running and time.monotonic() < deadline:
        time.sleep(0.005)

    status = manager.status()
    assert status["status"] == "failed"
    assert "双侧接触" in status["error"]
    assert status["holding_object"] is False
    assert gripper.positions[-1] == 1000
    assert [mode for mode, _ in robot.moves] == ["MoveJ_P", "MoveL", "MoveL"]


def test_updating_transfer_config_invalidates_old_preview() -> None:
    manager = TransferTaskManager(_Robot(), _Gripper(), _Sensors(), TransferConfig())
    plan = build_transfer_plan(
        _vision_result(), "gray_cube", TargetingConfig(), manager.config, _solution()
    )
    manager.set_preview(plan)

    status = manager.update_config(TransferConfig(pick_lift_mm=140.0))

    assert status["status"] == "idle"
    assert status["plan"] is None
    assert manager.config.pick_lift_mm == 140.0
