from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vtla_grip.calibration_solver import solve_calibration, solve_rigid_transform


def test_solve_rigid_transform_recovers_known_transform() -> None:
    source = np.asarray([[0, 0, 0], [100, 0, 0], [0, 100, 0], [20, 30, 80]], dtype=float)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.asarray([200, -50, 30], dtype=float)
    target = (rotation @ source.T).T + translation
    result = solve_rigid_transform(source, target)
    assert np.allclose(result[:3, :3], rotation, atol=1e-9)
    assert np.allclose(result[:3, 3], translation, atol=1e-9)


def test_calibration_report_rejects_one_outlier(tmp_path: Path) -> None:
    path = tmp_path / "blue-dot.jsonl"
    points = np.asarray(
        [
            [0, 0, 500],
            [80, 0, 520],
            [0, 90, 540],
            [70, 80, 560],
            [20, 30, 650],
            [100, 50, 610],
            [50, 120, 580],
            [130, 100, 680],
            [40, 160, 720],
            [150, 20, 740],
        ],
        dtype=float,
    )
    target = points + np.asarray([25, 300, -100])
    target[2] += 80
    events = [{"type": "session"}]
    for index, (camera, base) in enumerate(zip(points, target), 1):
        events.append(
            {
                "type": "sample",
                "sample_index": index,
                "camera_xyz_m": (camera / 1000).tolist(),
                "robot_tcp_pose_mm_deg": [*base.tolist(), 180, 0, 0],
            }
        )
    path.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
    report = solve_calibration(path, threshold_mm=15)
    assert report["inlier_count"] == 9
    assert report["outlier_sample_indices"] == [3]
    assert report["final_all_samples"]["median_mm"] < 1e-8


def test_rejects_degenerate_points() -> None:
    points = np.asarray([[value, 0, 0] for value in range(3)], dtype=float)
    with pytest.raises(ValueError, match="degenerate"):
        solve_rigid_transform(points, points)
