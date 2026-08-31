from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlaceTarget:
    pixel_uv: tuple[int, int]
    camera_xyz_m: tuple[float, float, float]
    base_xyz_mm: tuple[float, float, float]
    workspace_created_at: str
    calibration_created_at: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"configured": True, **asdict(self)}


def default_place_target_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "place_target.json"


class PlaceTargetStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_place_target_path()
        self._lock = threading.RLock()
        self._target = self._load()

    @property
    def target(self) -> PlaceTarget | None:
        with self._lock:
            return self._target

    def save(
        self,
        pixel_uv: tuple[int, int],
        camera_xyz_m: tuple[float, float, float],
        base_xyz_mm: tuple[float, float, float],
        workspace_created_at: str,
        calibration_created_at: str,
    ) -> dict[str, Any]:
        target = PlaceTarget(
            pixel_uv=pixel_uv,
            camera_xyz_m=camera_xyz_m,
            base_xyz_mm=base_xyz_mm,
            workspace_created_at=workspace_created_at,
            calibration_created_at=calibration_created_at,
            created_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"schema_version": 1, **asdict(target)}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
            self._target = target
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._target is None:
                return {"configured": False, "path": str(self.path.resolve())}
            return {**self._target.to_dict(), "path": str(self.path.resolve())}

    def _load(self) -> PlaceTarget | None:
        if not self.path.is_file():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != 1:
                return None
            return PlaceTarget(
                pixel_uv=tuple(map(int, raw["pixel_uv"])),
                camera_xyz_m=tuple(map(float, raw["camera_xyz_m"])),
                base_xyz_mm=tuple(map(float, raw["base_xyz_mm"])),
                workspace_created_at=str(raw["workspace_created_at"]),
                calibration_created_at=str(raw["calibration_created_at"]),
                created_at=str(raw["created_at"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
