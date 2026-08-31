import json

import pytest

from vtla_grip.calibration_app import CalibrationRecorder, prepare_robot_read_only
from vtla_grip.camera_app import ClickResult


class FakeRobotClient:
    def __init__(self, connected: bool = False) -> None:
        self.connected = connected
        self.calls: list[str] = []

    def ping(self):
        self.calls.append("ping")
        return {"robot": "RM65", "simulate": True}

    def status(self):
        self.calls.append("status")
        return {"connected": self.connected, "enabled": False}

    def connect(self):
        self.calls.append("connect")
        self.connected = True
        return {"connected": True, "enabled": False}

    def get_pose(self):
        self.calls.append("get_pose")
        return {"pose": [0.0] * 6, "joints": [0.0] * 6}


def test_prepare_robot_read_only_connects_without_enable_or_motion() -> None:
    client = FakeRobotClient()
    result = prepare_robot_read_only(client)  # type: ignore[arg-type]
    assert result["status"]["connected"] is True
    assert client.calls == ["ping", "status", "connect", "get_pose"]


def test_prepare_robot_read_only_reuses_existing_connection() -> None:
    client = FakeRobotClient(connected=True)
    prepare_robot_read_only(client)  # type: ignore[arg-type]
    assert client.calls == ["ping", "status", "get_pose"]


def test_recorder_writes_session_and_sample(tmp_path) -> None:
    output = tmp_path / "session.jsonl"
    recorder = CalibrationRecorder(output, {"camera": {"serial": "test"}})
    recorder.append_sample(
        ClickResult((20, 30), 0.5, (0.1, 0.2, 0.5)),
        {"pose": [1, 2, 3, 4, 5, 6], "joints": [10, 20, 30, 40, 50, 60]},
    )
    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "session"
    assert events[1]["type"] == "sample"
    assert events[1]["sample_index"] == 1
    assert events[1]["pixel_uv"] == [20, 30]
    assert events[1]["robot_tcp_pose_mm_deg"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_recorder_refuses_to_overwrite(tmp_path) -> None:
    output = tmp_path / "session.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        CalibrationRecorder(output, {})


def test_recorder_resumes_existing_samples(tmp_path) -> None:
    output = tmp_path / "session.jsonl"
    first = CalibrationRecorder(output, {"camera": {"serial": "test"}})
    first_sample = first.append_sample(
        ClickResult((20, 30), 0.5, (0.1, 0.2, 0.5)),
        {"pose": [1, 2, 3, 4, 5, 6], "joints": [10, 20, 30, 40, 50, 60]},
    )

    resumed = CalibrationRecorder.resume(output)
    assert resumed.sample_count == 1
    assert resumed.last_sample == first_sample
    second_sample = resumed.append_sample(
        ClickResult((40, 50), 0.6, (0.2, 0.3, 0.6)),
        {"pose": [7, 8, 9, 10, 11, 12], "joints": [11, 21, 31, 41, 51, 61]},
    )

    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert resumed.sample_count == 2
    assert second_sample["sample_index"] == 2
    assert [event["type"] for event in events] == ["session", "sample", "sample"]


def test_recorder_refuses_to_resume_malformed_record(tmp_path) -> None:
    output = tmp_path / "session.jsonl"
    output.write_text('{"type":"sample","schema_version":1,"sample_index":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="before session"):
        CalibrationRecorder.resume(output)
