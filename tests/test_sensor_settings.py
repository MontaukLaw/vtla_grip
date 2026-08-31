from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vtla_grip.config import load_config
from vtla_grip.hardware.signal_processing import GraspSuccessJudge, SensorSignalProcessor
from vtla_grip.sensor_settings import FIELDS, SensorSettingsStore


def test_migrated_schema_has_all_unique_arm_gripping_controls() -> None:
    paths = [field[2] for field in FIELDS]
    assert len(paths) == 97
    assert len(paths) == len(set(paths))
    assert "heatmap_display.preprocessing.release_gate.enabled" in paths
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
