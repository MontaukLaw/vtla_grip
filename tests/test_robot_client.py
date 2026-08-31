from dataclasses import replace

import pytest

from vtla_grip.config import RobotConfig
from vtla_grip.robot_client import RM65Client


def test_motion_is_locked_by_default() -> None:
    client = RM65Client(RobotConfig())
    with pytest.raises(PermissionError):
        client.move_to_pose([0, 0, 300, 180, 0, 0])


def test_motion_requires_six_pose_values() -> None:
    client = RM65Client(replace(RobotConfig(), allow_motion=True))
    with pytest.raises(ValueError):
        client.move_to_pose([0, 0, 300])
