from __future__ import annotations

import cv2
import numpy as np

from vtla_grip.config import VisionConfig
from vtla_grip.table_workspace import TableWorkspace, WorkspacePoint
from vtla_grip.vision import (
    TabletopVisionDetector,
    VisionDetection,
    build_auto_annotations,
    depth_object_proposals,
)


def _scene() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    depth = np.full((480, 640), 600, dtype=np.uint16)
    cv2.circle(image, (110, 150), 32, (0, 180, 0), thickness=-1)
    cv2.rectangle(image, (230, 120), (300, 190), (128, 128, 128), thickness=-1)
    cv2.rectangle(image, (300, 300), (500, 410), (0, 180, 0), thickness=-1)
    return image, depth


def test_detects_first_version_objects_and_green_tray() -> None:
    image, depth = _scene()
    detector = TabletopVisionDetector(VisionConfig())

    result = detector.detect(image, depth, 0.001)

    by_class = {item["class_name"]: item for item in result["detections"]}
    assert set(by_class) == {"green_cylinder", "gray_cube", "green_tray"}
    assert by_class["green_cylinder"]["graspable"] is True
    assert by_class["gray_cube"]["graspable"] is True
    assert by_class["green_tray"]["kind"] == "destination"
    assert result["recommended_object_id"] in {
        by_class["green_cylinder"]["detection_id"],
        by_class["gray_cube"]["detection_id"],
    }
    assert result["destination_id"] == by_class["green_tray"]["detection_id"]


def test_recommends_only_requested_object_class() -> None:
    detector = TabletopVisionDetector(VisionConfig())
    common = {
        "kind": "object",
        "confidence": 0.9,
        "bbox_xywh": (10, 10, 20, 20),
        "center_uv": (20, 20),
        "area_px": 400.0,
        "depth_m": 0.5,
        "depth_spread_m": 0.0,
        "depth_valid_fraction": 1.0,
        "graspable": True,
        "rejection_reason": None,
    }
    detections = [
        VisionDetection(detection_id="green", class_name="green_cylinder", **common),
        VisionDetection(detection_id="gray", class_name="gray_cube", **common),
    ]
    selected = detector.recommend_object(detections, "green_cylinder")
    assert selected is not None
    assert selected.class_name == "green_cylinder"


def test_invalid_depth_rejects_grasp_target() -> None:
    image, depth = _scene()
    depth[110:191, 70:151] = 0
    detector = TabletopVisionDetector(VisionConfig())

    result = detector.detect(image, depth, 0.001)

    cylinder = next(item for item in result["detections"] if item["class_name"] == "green_cylinder")
    assert cylinder["graspable"] is False
    assert cylinder["rejection_reason"] == "目标区域没有足够的有效深度"


def test_roi_excludes_objects_outside_the_table_region() -> None:
    image, depth = _scene()
    detector = TabletopVisionDetector(VisionConfig(roi_xyxy=(200, 80, 620, 450)))

    result = detector.detect(image, depth, 0.001)

    classes = {item["class_name"] for item in result["detections"]}
    assert classes == {"gray_cube", "green_tray"}


def test_polygon_workspace_excludes_objects_outside_its_edges() -> None:
    image, depth = _scene()
    detector = TabletopVisionDetector(VisionConfig())

    result = detector.detect(
        image,
        depth,
        0.001,
        roi_polygon_uv=[[200, 80], [620, 80], [610, 450], [210, 450]],
    )

    classes = {item["class_name"] for item in result["detections"]}
    assert classes == {"gray_cube", "green_tray"}
    assert result["roi_polygon_uv"] == [[200, 80], [620, 80], [610, 450], [210, 450]]


def test_auto_annotation_accepts_multiple_instances_of_selected_class() -> None:
    detector = TabletopVisionDetector(VisionConfig())
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    depth = np.full((480, 640), 600, dtype=np.uint16)
    cv2.circle(image, (140, 160), 30, (0, 180, 0), thickness=-1)
    cv2.circle(image, (310, 240), 34, (0, 180, 0), thickness=-1)

    result = detector.detect(image, depth, 0.001)
    annotation = build_auto_annotations(result, "green_cylinder", 640, 480)

    assert annotation["valid"] is True
    assert len(annotation["annotations"]) == 2
    assert all(item["class_id"] == 0 for item in annotation["annotations"])
    assert all(
        0.0 < value <= 1.0 for item in annotation["annotations"] for value in item["yolo_xywh"]
    )


def test_auto_annotation_rejects_mixed_classes() -> None:
    image, depth = _scene()
    result = TabletopVisionDetector(VisionConfig()).detect(image, depth, 0.001)

    annotation = build_auto_annotations(result, "green_cylinder", 640, 480)

    assert annotation["valid"] is False
    assert any("其他类别" in reason for reason in annotation["reasons"])


def test_empty_table_creates_valid_negative_annotation() -> None:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    depth = np.full((480, 640), 600, dtype=np.uint16)
    result = TabletopVisionDetector(VisionConfig()).detect(image, depth, 0.001)

    annotation = build_auto_annotations(result, "empty_table", 640, 480)

    assert annotation["valid"] is True
    assert annotation["annotations"] == []


def test_depth_proposals_find_unknown_object_above_table() -> None:
    points = tuple(
        WorkspacePoint(pixel_uv=uv, depth_m=1.0, camera_xyz_m=xyz, base_xyz_mm=(0, 0, 0))
        for uv, xyz in zip(
            [(10, 10), (90, 10), (90, 90), (10, 90)],
            [(-0.4, -0.4, 1.0), (0.4, -0.4, 1.0), (0.4, 0.4, 1.0), (-0.4, 0.4, 1.0)],
            strict=True,
        )
    )
    workspace = TableWorkspace(
        points=points,  # type: ignore[arg-type]
        plane_origin_camera_m=(0.0, 0.0, 1.0),
        plane_normal_camera=(0.0, 0.0, 1.0),
        plane_axis_u_camera=(1.0, 0.0, 0.0),
        plane_axis_v_camera=(0.0, 1.0, 0.0),
        polygon_plane_xy_m=((-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)),
        max_plane_residual_m=0.0,
        created_at="test",
    )
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    depth[35:66, 40:71] = 900

    result = depth_object_proposals(
        depth,
        0.001,
        {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
        workspace,
        VisionConfig(proposal_min_area_px=100),
    )

    assert len(result["boxes"]) == 1
    x, y, width, height = result["boxes"][0]
    assert x <= 40 and y <= 35
    assert x + width >= 70 and y + height >= 65


def test_gray_cube_uses_local_contrast_when_table_is_also_low_saturation() -> None:
    image = np.full((480, 640, 3), 205, dtype=np.uint8)
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    image[210:260, 300:345] = (125, 125, 125)
    depth[210:260, 300:345] = 920
    detector = TabletopVisionDetector(VisionConfig())

    result = detector.detect(image, depth, 0.001)

    cube = next(item for item in result["detections"] if item["class_name"] == "gray_cube")
    assert cube["bbox_xywh"] == (300, 210, 45, 50)
    assert cube["graspable"] is True


def test_realistic_projected_green_tray_is_not_a_cylinder() -> None:
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    depth = np.full((480, 640), 900, dtype=np.uint16)
    polygon = np.asarray([[310, 250], [380, 240], [395, 290], [325, 305]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (0, 135, 65))
    detector = TabletopVisionDetector(VisionConfig())

    result = detector.detect(image, depth, 0.001)

    green = [item for item in result["detections"] if item["class_name"].startswith("green_")]
    assert len(green) == 1
    assert green[0]["class_name"] == "green_tray"
    assert green[0]["kind"] == "destination"
    assert result["destination_id"] == green[0]["detection_id"]
