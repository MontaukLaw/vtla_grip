from __future__ import annotations

import json
from dataclasses import replace

from vtla_grip.camera_app import ClickResult
from vtla_grip.config import CalibrationConfig, TargetingConfig, load_config
from vtla_grip.web.app import ControlState, _target_preview, _validate_motion_target


def test_target_preview_applies_marker_offset(tmp_path) -> None:
    solution_path = tmp_path / "calibration.json"
    solution_path.write_text(
        json.dumps(
            {
                "transform_4x4": [
                    [1, 0, 0, 10],
                    [0, 1, 0, 20],
                    [0, 0, 1, 30],
                    [0, 0, 0, 1],
                ],
                "fixed_tool_orientation_deg": [180, 0, -5],
                "tool_orientation_range_deg": [0, 0, 0],
                "quality": {"hover_ready": True},
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        load_config(),
        calibration=CalibrationConfig(output_path=str(solution_path)),
        targeting=TargetingConfig(
            marker_to_tip_xyz_mm=(15, -25, 400),
        ),
    )
    state = ControlState(config)
    state.selected_point = ClickResult((100, 200), 0.5, (0.1, 0.2, 0.5))

    preview = _target_preview(state)

    assert preview["target_pose_mm_deg"] == [125.0, 195.0, 930.0, 180.0, 0.0, -5.0]


def test_motion_target_has_no_application_distance_limit() -> None:
    state = ControlState(load_config())

    _validate_motion_target(
        state,
        [10_000.0, -10_000.0, 5_000.0, 180.0, 0.0, 0.0],
        speed=5,
    )
