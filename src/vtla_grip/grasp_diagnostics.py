from __future__ import annotations

import time
from typing import Any

from .hardware.signal_processing import bilateral_contact_ready


def closure_snapshot(gripper: Any, data: dict[str, Any], command: int, minimum: int) -> dict[str, Any]:
    """Read actual position without making diagnostic failures change grasp decisions."""
    started = time.monotonic()
    actual = None
    error = None
    try:
        value = gripper.read_position() if hasattr(gripper, "read_position") else gripper.feedback().position
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000:
            actual = value
        else:
            error = f"无效位置反馈：{value!r}"
    except Exception as exc:  # noqa: BLE001 - diagnostics must not abort closure.
        error = str(exc)
    grasp = data.get("grasp_success") or {}
    features = data.get("features") or {}
    return {
        "position_command": command,
        "actual_position": actual,
        "position_error": actual - command if actual is not None else None,
        "minimum_position": minimum,
        "position_read_error": error,
        "position_read_ms": round((time.monotonic() - started) * 1000, 1),
        "frame_ready": data.get("frame_ready"),
        "sensor_frame_ids": data.get("frame_ids"),
        "left_baseline_ready": (features.get("left") or {}).get("baseline_ready"),
        "right_baseline_ready": (features.get("right") or {}).get("baseline_ready"),
        **{key: grasp.get(key) for key in (
            "left_peak", "right_peak", "left_threshold", "right_threshold",
            "both_sides_over_threshold", "confirm_count", "required_frames", "success",
        )},
        "contact_accepted": bilateral_contact_ready(data),
    }
