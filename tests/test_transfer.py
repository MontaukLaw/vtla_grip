from __future__ import annotations

import time

import pytest

from vtla_grip.config import TargetingConfig, TransferConfig
from vtla_grip.transfer import (
    TransferError,
    TransferTaskManager,
    build_transfer_plan,
)


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
        self.zero_count = getattr(self, "zero_count", 0) + 1
        return {}

    def data(self) -> dict:
        self.reads += 1
        return {
            "frame_ready": True,
            "features": {
                "left": {"baseline_ready": True},
                "right": {"baseline_ready": True},
            },
            "grasp_success": {
                "both_sides_over_threshold": self.contact_after is not None and self.reads >= self.contact_after,
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
    assert manager.sensors.zero_count >= 2


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
            wait_for_arrival_feedback=True,
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


def test_move_does_not_poll_arrival_feedback_by_default() -> None:
    class CountingRobot(_Robot):
        def __init__(self) -> None:
            super().__init__()
            self.pose_reads = 0

        def pose(self) -> dict:
            self.pose_reads += 1
            return super().pose()

    robot = CountingRobot()
    manager = TransferTaskManager(robot, _Gripper(), _Sensors(), TransferConfig())

    manager._move([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "MoveL", 5, {
        "acceleration": 10,
    })

    assert robot.pose_reads == 0


def test_no_contact_releases_after_command_sequence() -> None:
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

    manager._thread.join(timeout=1)
    status = manager.status()
    assert status["status"] == "failed"
    assert "双侧接触" in (status["error"] or "")
    assert status["holding_object"] is False
    assert gripper.positions[-1] == manager.config.gripper_open_position
    assert [mode for mode, _ in robot.moves] == ["MoveJ_P", "MoveL", "MoveL"]


def test_no_contact_sends_each_close_position_once():
    gripper = _Gripper()
    manager = TransferTaskManager(
        _Robot(), gripper, _Sensors(contact_after=None),
        TransferConfig(gripper_min_position=700, gripper_close_step=100, grasp_poll_interval_s=0),
    )
    with pytest.raises(Exception, match="双侧接触"):
        manager._grasp_until_contact()
    assert gripper.positions == [900, 800, 700]


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


def _two_candidate_plan(config, target_property="软"):
    vision = _vision_result()
    first = {**vision["recommended_object"], "graspable": True}
    second = {
        **first, "detection_id": "object-2", "base_xyz_mm": [60.0, 20.0, 5.0],
        "height_above_table_mm": 55.0,
    }
    vision["detections"] = [
        first, second, {**second, "detection_id": "blocked", "graspable": False},
        {**second, "detection_id": "other", "class_name": "green_cylinder"},
    ]
    return build_transfer_plan(
        vision, "gray_cube", TargetingConfig(), config, _solution(),
        target_property=target_property,
    )


@pytest.mark.parametrize("properties,expected_status,attempts", [
    (["软"], "completed", 1),
    (["硬", "软"], "completed", 2),
    (["硬", "硬"], "failed", 2),
    (["error"], "failed", 1),
])
def test_property_selection_only_transports_verified_match(properties, expected_status, attempts):
    class SequenceRecognizer(_Recognizer):
        def predict(self, vision_class, frames, final_gripper_position):
            label = properties.pop(0)
            if label == "error":
                raise RuntimeError("测试识别失败")
            result = super().predict(vision_class, frames, final_gripper_position)
            result.property = label
            return result

    robot = _Robot()
    releases = []

    class RecordingGripper(_Gripper):
        def set_position(self, value):
            if value == 1000:
                releases.append(robot.moves[-1][1] if robot.moves else None)
            return super().set_position(value)

    gripper = RecordingGripper()
    config = TransferConfig(
        release_wait_s=0, grasp_poll_interval_s=0, gripper_close_step=100,
        arrival_stable_frames=1,
    )
    manager = TransferTaskManager(robot, gripper, _Sensors(), config, recognizer=SequenceRecognizer())
    plan = _two_candidate_plan(config)
    assert len(plan["candidates"]) == 2
    assert plan["command"]["target_property"] == "软"
    manager.set_preview(plan)
    manager._run(plan)
    status = manager.status()
    assert status["status"] == expected_status
    assert not status["holding_object"]
    assert len(status["attempts"]) == attempts
    assert not properties
    assert gripper.positions[-1] == config.gripper_open_position
    if attempts == 2:
        assert releases[1] == plan["candidates"][0]["poses"]["pick"]
        # Release at original pick height, retreat vertically before approaching candidate 2.
        assert robot.moves[2] == ("MoveL", plan["candidates"][0]["poses"]["pick_above"])
        assert robot.moves[3] == ("MoveJ_P", plan["candidates"][1]["poses"]["pick_above"])
    if expected_status == "completed":
        selected = plan["candidates"][attempts - 1]
        assert robot.moves[-2] == ("MoveL", selected["poses"]["place"])
        assert status["attempts"][-1]["matched"]
    else:
        assert len(robot.moves) == 3 * attempts
        assert releases[-1] == plan["candidates"][attempts - 1]["poses"]["pick"]
        assert status["error"]


def test_invalid_property_is_rejected():
    with pytest.raises(TransferError, match="软硬要求"):
        _two_candidate_plan(TransferConfig(), "未知")


def test_unrestricted_plan_keeps_single_recommended_object():
    plan = _two_candidate_plan(TransferConfig(), None)
    assert "candidates" not in plan
    assert "target_property" not in plan["command"]


def test_stop_during_recognition_preserves_grip_and_does_not_try_next_candidate():
    robot, gripper = _Robot(), _Gripper()
    config = TransferConfig(
        release_wait_s=0, grasp_poll_interval_s=0, arrival_stable_frames=1,
    )

    class StoppingRecognizer(_Recognizer):
        def predict(self, vision_class, frames, final_gripper_position):
            manager.request_stop()
            result = super().predict(vision_class, frames, final_gripper_position)
            result.property = "硬"
            return result

    manager = TransferTaskManager(robot, gripper, _Sensors(), config, recognizer=StoppingRecognizer())
    plan = _two_candidate_plan(config)
    manager.set_preview(plan)
    manager._run(plan)
    assert manager.status()["status"] == "stopped"
    assert manager.status()["holding_object"]
    assert len(robot.moves) == 2
    assert gripper.positions[-1] < config.gripper_open_position
