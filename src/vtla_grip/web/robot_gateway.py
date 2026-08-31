from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..config import RobotConfig
from ..robot_client import RM65Client, RobotServiceError


class RobotGateway:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.client = RM65Client(config)
        self._process: subprocess.Popen[bytes] | None = None

    def start_service(self, simulate: bool = False) -> dict[str, Any]:
        try:
            return {"started_by_app": False, **self.client.ping()}
        except (OSError, RobotServiceError):
            pass
        script = Path(self.config.service_script)
        if not script.is_file():
            raise FileNotFoundError(f"RM65 service script not found: {script}")
        command = [sys.executable, str(script)]
        if simulate:
            command.append("--simulate")
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            command,
            cwd=script.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + 8.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return {"started_by_app": True, **self.client.ping()}
            except (OSError, RobotServiceError) as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"RM65 service did not become ready: {last_error}")

    def status(self, include_pose: bool = False) -> dict[str, Any]:
        try:
            status = self.client.status()
            result = {
                "service_online": True,
                "motion_configured": self.config.allow_motion,
                **status,
            }
            if include_pose and status.get("connected"):
                try:
                    result.update(self.client.get_pose())
                except (OSError, RobotServiceError) as exc:
                    result["pose_error"] = str(exc)
            return result
        except (OSError, RobotServiceError) as exc:
            return {
                "service_online": False,
                "connected": False,
                "enabled": False,
                "motion_active": False,
                "motion_configured": self.config.allow_motion,
                "error": str(exc),
            }

    def connect(self) -> dict[str, Any]:
        return self.client.connect()

    def enable(self) -> dict[str, Any]:
        return self.client.request("enable")

    def disconnect(self) -> dict[str, Any]:
        return self.client.request("disconnect")

    def disable(self) -> dict[str, Any]:
        return self.client.request("disable")

    def slow_stop(self) -> dict[str, Any]:
        return self.client.request("slow_stop")

    def pose(self) -> dict[str, Any]:
        return self.client.get_pose()

    def move(self, pose: list[float], mode: str, speed: int, acc: int) -> dict[str, Any]:
        if not self.config.allow_motion:
            raise PermissionError("robot motion is disabled in config/default.json")
        return self.client.request(
            "move_to_pose",
            response_timeout_s=self.config.motion_timeout_s,
            pose=pose,
            mode=mode,
            speed=speed,
            acc=acc,
        )
