from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

POINT_LABELS = ("左上", "右上", "右下", "左下")


@dataclass(frozen=True)
class WorkspacePoint:
    pixel_uv: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]
    base_xyz_mm: tuple[float, float, float]


@dataclass(frozen=True)
class TableWorkspace:
    points: tuple[WorkspacePoint, WorkspacePoint, WorkspacePoint, WorkspacePoint]
    plane_origin_camera_m: tuple[float, float, float]
    plane_normal_camera: tuple[float, float, float]
    plane_axis_u_camera: tuple[float, float, float]
    plane_axis_v_camera: tuple[float, float, float]
    polygon_plane_xy_m: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    max_plane_residual_m: float
    created_at: str

    @property
    def polygon_uv(self) -> list[list[int]]:
        return [list(point.pixel_uv) for point in self.points]

    def contains_camera_point(self, camera_xyz_m: list[float] | tuple[float, ...]) -> bool:
        point = np.asarray(camera_xyz_m, dtype=np.float64)
        origin = np.asarray(self.plane_origin_camera_m, dtype=np.float64)
        axis_u = np.asarray(self.plane_axis_u_camera, dtype=np.float64)
        axis_v = np.asarray(self.plane_axis_v_camera, dtype=np.float64)
        relative = point - origin
        projected = (float(relative @ axis_u), float(relative @ axis_v))
        polygon = np.asarray(self.polygon_plane_xy_m, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, projected, False) >= 0

    def height_above_plane_m(self, camera_xyz_m: list[float] | tuple[float, ...]) -> float:
        point = np.asarray(camera_xyz_m, dtype=np.float64)
        origin = np.asarray(self.plane_origin_camera_m, dtype=np.float64)
        normal = np.asarray(self.plane_normal_camera, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if float(normal @ -origin) < 0.0:
            normal = -normal
        return max(0.0, float((point - origin) @ normal))

    def intersect_camera_ray_with_plane(
        self, camera_ray_point_m: list[float] | tuple[float, ...]
    ) -> tuple[float, float, float]:
        """Intersect the camera-origin ray through a deprojected pixel with the table plane."""
        direction = np.asarray(camera_ray_point_m, dtype=np.float64)
        origin = np.asarray(self.plane_origin_camera_m, dtype=np.float64)
        normal = np.asarray(self.plane_normal_camera, dtype=np.float64)
        denominator = float(normal @ direction)
        if abs(denominator) < 1e-9:
            raise ValueError("鼠标射线与桌面平面平行，无法确定放置点")
        distance_scale = float(normal @ origin) / denominator
        if not np.isfinite(distance_scale) or distance_scale <= 0.0:
            raise ValueError("鼠标射线没有与相机前方的桌面相交")
        intersection = direction * distance_scale
        if not self.contains_camera_point(intersection):
            raise ValueError("指定放置点必须位于四点桌面工作区内")
        return tuple(map(float, intersection))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": self.created_at,
            "points": [asdict(point) for point in self.points],
            "plane_origin_camera_m": list(self.plane_origin_camera_m),
            "plane_normal_camera": list(self.plane_normal_camera),
            "plane_axis_u_camera": list(self.plane_axis_u_camera),
            "plane_axis_v_camera": list(self.plane_axis_v_camera),
            "polygon_plane_xy_m": [list(point) for point in self.polygon_plane_xy_m],
            "max_plane_residual_m": self.max_plane_residual_m,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TableWorkspace:
        points = tuple(
            WorkspacePoint(
                pixel_uv=tuple(item["pixel_uv"]),
                depth_m=float(item["depth_m"]),
                camera_xyz_m=tuple(item["camera_xyz_m"]),
                base_xyz_mm=tuple(item["base_xyz_mm"]),
            )
            for item in raw["points"]
        )
        if len(points) != 4:
            raise ValueError("table workspace must contain exactly four points")
        return cls(
            points=points,  # type: ignore[arg-type]
            plane_origin_camera_m=tuple(raw["plane_origin_camera_m"]),
            plane_normal_camera=tuple(raw["plane_normal_camera"]),
            plane_axis_u_camera=tuple(raw["plane_axis_u_camera"]),
            plane_axis_v_camera=tuple(raw["plane_axis_v_camera"]),
            polygon_plane_xy_m=tuple(tuple(item) for item in raw["polygon_plane_xy_m"]),  # type: ignore[arg-type]
            max_plane_residual_m=float(raw["max_plane_residual_m"]),
            created_at=str(raw["created_at"]),
        )


def fit_table_workspace(
    points: list[WorkspacePoint], *, maximum_residual_m: float = 0.015
) -> TableWorkspace:
    if len(points) != 4:
        raise ValueError("桌面工作区需要按顺序选择四个角")
    pixels = np.asarray([point.pixel_uv for point in points], dtype=np.int32)
    if abs(float(cv2.contourArea(pixels))) < 5_000.0:
        raise ValueError("桌面工作区画面面积过小")
    if not cv2.isContourConvex(pixels):
        raise ValueError("四个角必须按左上、右上、右下、左下顺序形成凸四边形")

    camera = np.asarray([point.camera_xyz_m for point in points], dtype=np.float64)
    origin = camera.mean(axis=0)
    centered = camera - origin
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    residuals = np.abs(centered @ normal)
    maximum_residual = float(residuals.max())
    if maximum_residual > maximum_residual_m:
        raise ValueError(
            f"四点不在同一桌面平面：最大残差 {maximum_residual * 1000:.1f} mm，"
            f"允许值 {maximum_residual_m * 1000:.1f} mm"
        )

    top_edge = camera[1] - camera[0]
    axis_u = top_edge - normal * float(top_edge @ normal)
    axis_u_norm = float(np.linalg.norm(axis_u))
    if axis_u_norm < 1e-6:
        raise ValueError("桌面上边界长度无效")
    axis_u /= axis_u_norm
    axis_v = np.cross(normal, axis_u)
    bottom_direction = (camera[2] + camera[3]) * 0.5 - (camera[0] + camera[1]) * 0.5
    if float(axis_v @ bottom_direction) < 0:
        normal = -normal
        axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    plane_xy = tuple((float(relative @ axis_u), float(relative @ axis_v)) for relative in centered)
    return TableWorkspace(
        points=tuple(points),  # type: ignore[arg-type]
        plane_origin_camera_m=tuple(map(float, origin)),
        plane_normal_camera=tuple(map(float, normal)),
        plane_axis_u_camera=tuple(map(float, axis_u)),
        plane_axis_v_camera=tuple(map(float, axis_v)),
        polygon_plane_xy_m=plane_xy,  # type: ignore[arg-type]
        max_plane_residual_m=maximum_residual,
        created_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
    )


def default_workspace_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "table_workspace.json"


class TableWorkspaceStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_workspace_path()
        self._lock = threading.RLock()
        self._draft: list[WorkspacePoint] = []
        self._active = False
        self._workspace = self._load()

    @property
    def workspace(self) -> TableWorkspace | None:
        with self._lock:
            return self._workspace

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._draft = []
            self._active = True
            return self.status()

    def add_point(self, point: WorkspacePoint) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                raise ValueError("请先开始桌面工作区标定")
            if len(self._draft) >= 4:
                raise ValueError("桌面工作区已经选择四个点")
            self._draft.append(point)
            if len(self._draft) == 4:
                try:
                    workspace = fit_table_workspace(self._draft)
                    self._save(workspace)
                    self._workspace = workspace
                    self._active = False
                    self._draft = []
                except ValueError:
                    self._draft.pop()
                    raise
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            workspace = self._workspace
            next_index = len(self._draft)
            return {
                "calibrated": workspace is not None,
                "active": self._active,
                "point_count": len(self._draft),
                "next_label": POINT_LABELS[next_index] if self._active and next_index < 4 else None,
                "draft_points": [asdict(point) for point in self._draft],
                "polygon_uv": workspace.polygon_uv if workspace else None,
                "max_plane_residual_mm": (
                    workspace.max_plane_residual_m * 1000.0 if workspace else None
                ),
                "created_at": workspace.created_at if workspace else None,
                "path": str(self.path.resolve()),
            }

    def _load(self) -> TableWorkspace | None:
        if not self.path.exists():
            return None
        try:
            return TableWorkspace.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _save(self, workspace: TableWorkspace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(workspace.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
