from __future__ import annotations

import argparse
import json
import socket
import uuid
from dataclasses import dataclass
from typing import Any

from .config import RobotConfig, load_config


class RobotServiceError(RuntimeError):
    pass


@dataclass
class RM65Client:
    config: RobotConfig
    timeout_s: float = 5.0

    def request(
        self,
        command: str,
        *,
        response_timeout_s: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        payload = {"id": uuid.uuid4().hex, "command": command, **params}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        with socket.create_connection(
            (self.config.host, self.config.port), timeout=self.timeout_s
        ) as connection:
            connection.settimeout(response_timeout_s or self.timeout_s)
            connection.sendall(encoded)
            response_file = connection.makefile("rb")
            raw = response_file.readline(64 * 1024 + 1)
        if not raw or len(raw) > 64 * 1024:
            raise RobotServiceError("invalid response from RM65 service")
        response = json.loads(raw.decode("utf-8"))
        if response.get("id") != payload["id"]:
            raise RobotServiceError("RM65 response id mismatch")
        if not response.get("ok"):
            raise RobotServiceError(str(response.get("error", "unknown RM65 service error")))
        return response["result"]

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def status(self) -> dict[str, Any]:
        return self.request("status")

    def connect(self) -> dict[str, Any]:
        """Connect the service to RM65 without enabling any joint."""
        return self.request("connect")

    def get_pose(self) -> dict[str, Any]:
        return self.request("get_pose")

    def move_to_pose(
        self,
        pose_mm_deg: list[float],
        mode: str = "MoveJ_P",
        speed: int | None = None,
        acc: int = 10,
    ) -> dict[str, Any]:
        if not self.config.allow_motion:
            raise PermissionError(
                "robot motion is locked; set robot.allow_motion=true only after calibration and safety checks"
            )
        if len(pose_mm_deg) != 6:
            raise ValueError("pose must contain [X,Y,Z,RX,RY,RZ]")
        return self.request(
            "move_to_pose",
            response_timeout_s=self.config.motion_timeout_s,
            pose=[float(value) for value in pose_mm_deg],
            mode=mode,
            speed=speed or self.config.default_speed,
            acc=acc,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only RM65 service diagnostics")
    parser.add_argument("command", choices=["ping", "status", "pose"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    client = RM65Client(load_config(args.config).robot)
    command = "get_pose" if args.command == "pose" else args.command
    print(json.dumps(client.request(command), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
