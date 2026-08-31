from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vtla_grip.table_workspace import (
    TableWorkspaceStore,
    WorkspacePoint,
    fit_table_workspace,
)


def _point(pixel: tuple[int, int], x: float, y: float) -> WorkspacePoint:
    z = 0.9 + 0.18 * x - 0.11 * y
    return WorkspacePoint(
        pixel_uv=pixel,
        depth_m=z,
        camera_xyz_m=(x, y, z),
        base_xyz_mm=(x * 1000.0, y * 1000.0, 200.0),
    )


def _points() -> list[WorkspacePoint]:
    return [
        _point((80, 70), -0.30, -0.20),
        _point((560, 90), 0.30, -0.20),
        _point((530, 420), 0.30, 0.25),
        _point((100, 400), -0.30, 0.25),
    ]


def test_intersects_camera_ray_with_tilted_table_plane() -> None:
    workspace = fit_table_workspace(_points())
    expected = np.asarray([0.1, 0.05, 0.9125])
    ray_point = expected * 0.72

    intersection = workspace.intersect_camera_ray_with_plane(ray_point)

    np.testing.assert_allclose(intersection, expected, atol=1e-9)


def test_rejects_place_ray_outside_workspace() -> None:
    workspace = fit_table_workspace(_points())
    outside = np.asarray([0.8, 0.8, 0.956])
    with pytest.raises(ValueError, match="工作区内"):
        workspace.intersect_camera_ray_with_plane(outside)


def test_fits_tilted_table_and_contains_projected_object() -> None:
    workspace = fit_table_workspace(_points())

    assert workspace.max_plane_residual_m < 1e-9
    assert workspace.contains_camera_point((0.0, 0.0, 0.75)) is True
    assert workspace.contains_camera_point((0.6, 0.0, 0.75)) is False

    origin = np.asarray(workspace.plane_origin_camera_m)
    normal = np.asarray(workspace.plane_normal_camera)
    if float(normal @ -origin) < 0.0:
        normal = -normal
    point_above = origin + normal * 0.04
    assert workspace.height_above_plane_m(tuple(point_above)) == pytest.approx(0.04)


def test_rejects_non_coplanar_fourth_point() -> None:
    points = _points()
    points[-1] = replace(
        points[-1],
        camera_xyz_m=(points[-1].camera_xyz_m[0], points[-1].camera_xyz_m[1], 1.25),
    )

    with pytest.raises(ValueError, match="不在同一桌面平面"):
        fit_table_workspace(points)


def test_store_keeps_last_valid_workspace_until_new_four_points_succeed(tmp_path) -> None:
    path = tmp_path / "table_workspace.json"
    store = TableWorkspaceStore(path)
    store.start()
    for point in _points():
        status = store.add_point(point)

    assert status["calibrated"] is True
    assert status["active"] is False
    assert path.exists()

    reloaded = TableWorkspaceStore(path)
    assert reloaded.workspace is not None
    assert np.allclose(reloaded.workspace.points[0].camera_xyz_m, _points()[0].camera_xyz_m)

    status = reloaded.start()
    assert status["calibrated"] is True
    assert status["active"] is True
    assert status["next_label"] == "左上"
