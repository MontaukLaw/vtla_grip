from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vtla_grip.config import load_config
from vtla_grip.hardware.signal_processing import GraspSuccessJudge, SensorSignalProcessor
from vtla_grip.sensor_settings import FIELDS, SensorSettingsStore


def test_migrated_schema_has_all_unique_arm_gripping_controls() -> None:
    paths = [field[2] for field in FIELDS]
    assert len(paths) == 96
    assert len(paths) == len(set(paths))
    assert "heatmap_display.preprocessing.release_gate.enabled" in paths
    assert "heatmap_display.preprocessing.release_gate.release_level" in paths
    assert "heatmap_display.preprocessing.release_gate.release_drop_abs" not in paths
    assert "state_transitions.contact_stop_trigger.point_threshold" in paths
    assert "adaptive_grasper.release_zero_frames" in paths
    assert "point_cloud.dense_rows" in paths


def test_sensor_store_updates_only_requested_file(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "config" / "sensor_processing.json"
    destination = tmp_path / "sensor_processing.json"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    store = SensorSettingsStore(destination)

    result = store.update("left", {"heatmap_display.preprocessing.signal_mask": 12.5})

    assert result["values"]["left"]["heatmap_display"]["preprocessing"]["signal_mask"] == 12.5
    assert result["values"]["right"]["heatmap_display"]["preprocessing"]["signal_mask"] == 3.0
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert (
        persisted["sensor_type_overrides"]["left"]["heatmap_display"]["preprocessing"][
            "signal_mask"
        ]
        == 12.5
    )


def test_signal_processor_applies_masks_without_baseline() -> None:
    config = {
        "heatmap_display": {
            "baseline": {"enabled": False},
            "std_filter": {"enabled": False},
            "preprocessing": {
                "signal_mask": 3.0,
                "signal_high_mask": 20.0,
                "final_signal_mask": 10.0,
                "median_filter": {"enabled": False, "window": 3},
                "release_gate": {"enabled": False},
            },
            "temporal_filter": {"enabled": False, "window": 3},
            "dynamic_threshold": {"enabled": False},
            "display": {"threshold": 100.0},
        }
    }
    processor = SensorSignalProcessor(config)

    processed, display, state = processor.process([2.0, 5.0, 15.0, 25.0] + [0.0] * 28)

    assert processed[:4] == [0.0, 5.0, 15.0, 0.0]
    assert display[:4] == [0.0, 0.0, 15.0, 0.0]
    assert state["baseline_ready"] is True


def test_release_gate_requires_value_below_minimum_release_value() -> None:
    config = {
        "heatmap_display": {
            "baseline": {"enabled": False},
            "std_filter": {"enabled": False},
            "preprocessing": {
                "release_gate": {
                    "enabled": True,
                    "release_slope": 0.3,
                    "release_level": 50.0,
                    "release_frames": 1,
                }
            },
        }
    }
    processor = SensorSignalProcessor(config)
    previous = np.full(32, 100.0)
    processor._apply_release_gate(previous)

    above_minimum = processor._apply_release_gate(np.full(32, 60.0))
    assert above_minimum[0] == 60.0

    below_minimum = processor._apply_release_gate(np.full(32, 40.0))
    assert below_minimum[0] == 0.0


def test_grasp_success_uses_both_sides_and_strict_confirmation() -> None:
    left_config = {
        "state_transitions": {
            "contact_stop_trigger": {"point_threshold": 10, "contact_confirm_frames": 1}
        }
    }
    right_config = {
        "state_transitions": {
            "contact_stop_trigger": {"point_threshold": 5, "contact_confirm_frames": 0}
        }
    }
    judge = GraspSuccessJudge()

    first = judge.update(np.asarray([11.0]), np.asarray([6.0]), left_config, right_config)
    second = judge.update(np.asarray([11.0]), np.asarray([6.0]), left_config, right_config)

    assert first["success"] is False
    assert second["success"] is True
    assert second["required_frames"] == 2


def test_robot_motion_speed_limit_is_100() -> None:
    assert load_config().targeting.maximum_speed == 100


def test_single_side_contact_never_confirms_grasp():
    config = {"state_transitions": {"contact_stop_trigger": {
        "point_threshold": 10, "contact_confirm_frames": 2,
    }}}
    judge = GraspSuccessJudge()
    for _ in range(5):
        assert not judge.update(np.array([100]), np.array([0]), config, config)["success"]
    for index in range(3):
        result = judge.update(np.array([100]), np.array([100]), config, config)
        assert result["success"] == (index == 2)
    assert not judge.update(np.array([0]), np.array([100]), config, config)["success"]


def test_capture_gate_rejects_stale_or_unilateral_contact():
    from vtla_grip.hardware.signal_processing import bilateral_contact_ready
    data = {
        "frame_ready": True,
        "features": {"left": {"baseline_ready": True}, "right": {"baseline_ready": True}},
        "grasp_success": {"success": True, "both_sides_over_threshold": False},
    }
    assert not bilateral_contact_ready(data)
    data["grasp_success"]["both_sides_over_threshold"] = True
    assert bilateral_contact_ready(data)
    data["frame_ready"] = False
    assert not bilateral_contact_ready(data)


def test_manual_zero_removes_constant_offset(tmp_path):
    from types import SimpleNamespace

    from vtla_grip.hardware.infineon import DualInfineonSensors
    config = {"heatmap_display": {"baseline": {"enabled": False}, "std_filter": {"enabled": False}}}
    sensors = DualInfineonSensors(load_config().sensors, SimpleNamespace(side=lambda side: config))

    class Reader:
        def __init__(self, value):
            self.value, self.id = value, 1
        def snapshot(self):
            return [self.value] * 32
        def frame_id(self):
            return self.id
        def status(self, stale_after_s):
            return {"connected": True, "frame_ready": True}

    sensors.left, sensors.right = Reader(100), Reader(200)
    assert max(sensors.data()["processed"]) == 200
    zeroed = sensors.zero()
    assert max(zeroed["processed"]) == 0
    assert zeroed["zeroed_at"] is not None
    sensors.left.value, sensors.right.value = 120, 220
    sensors.left.id += 1
    assert sensors.data()["frame_ids"] == [1, 1]
    sensors.right.id += 1
    assert sensors.data()["frame_ids"] == [2, 2]
    assert max(sensors.data()["processed"]) == 20
