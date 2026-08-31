from __future__ import annotations

import time

from vtla_grip.config import TransferConfig
from vtla_grip.tactile_capture import TactileCaptureManager
from vtla_grip.tactile_recognition import TactileRecognitionWorkspace


class _Gripper:
    def __init__(self) -> None:
        self.positions: list[int] = []

    def set_force(self, value: int) -> None:
        return None

    def set_speed(self, value: int) -> None:
        return None

    def set_position(self, value: int) -> None:
        self.positions.append(value)

    def feedback(self):
        return type("Feedback", (), {"position": 420})()


class _Sensors:
    def __init__(self) -> None:
        self.reads = 0

    def zero(self) -> None:
        return None

    def data(self) -> dict:
        self.reads += 1
        return {
            "features": {
                "left": {"baseline_ready": True},
                "right": {"baseline_ready": True},
            },
            "processed": [float(self.reads)] * 64,
            "grasp_success": {"success": self.reads >= 2},
        }


def test_manual_capture_requires_explicit_save(tmp_path) -> None:
    workspace = TactileRecognitionWorkspace(tmp_path / "data", tmp_path / "models")
    workspace.sync_vision_classes([{"name": "cube"}])
    gripper = _Gripper()
    manager = TactileCaptureManager(
        gripper,
        _Sensors(),
        workspace,
        TransferConfig(
            release_wait_s=0.001,
            grasp_poll_interval_s=0.001,
            gripper_close_step=100,
        ),
    )

    manager.start("cube", "软")
    deadline = time.monotonic() + 2.0
    while manager.running and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.status()["status"] == "ready"
    assert workspace.list_samples()["total"] == 0
    saved = manager.save("cube", "软")
    assert saved["status"] == "saved"
    assert workspace.list_samples()["total"] == 1
    assert gripper.positions[-1] == manager.config.gripper_open_position
