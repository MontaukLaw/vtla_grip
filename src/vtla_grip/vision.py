from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .config import VisionConfig
from .table_workspace import TableWorkspace

YOLO_CLASSES = ("green_cylinder", "gray_cube", "green_tray")
AUTO_CAPTURE_CLASSES = (*YOLO_CLASSES, "empty_table")


def depth_object_proposals(
    depth_raw: np.ndarray,
    depth_scale: float,
    intrinsics: dict[str, Any],
    workspace: TableWorkspace,
    config: VisionConfig,
) -> dict[str, Any]:
    """Find class-agnostic tabletop objects from their height above the fitted plane."""
    if depth_raw.ndim != 2:
        raise ValueError("depth frame must be a single-channel image")
    height, width = depth_raw.shape
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    if fx <= 0 or fy <= 0:
        raise ValueError("camera intrinsics are invalid")

    ys, xs = np.indices((height, width), dtype=np.float32)
    z = depth_raw.astype(np.float32) * float(depth_scale)
    x = (xs - ppx) / fx * z
    y = (ys - ppy) / fy * z
    origin = np.asarray(workspace.plane_origin_camera_m, dtype=np.float32)
    normal = np.asarray(workspace.plane_normal_camera, dtype=np.float32)
    if float(normal @ -origin) < 0.0:
        normal = -normal
    plane_height = (
        (x - origin[0]) * normal[0] + (y - origin[1]) * normal[1] + (z - origin[2]) * normal[2]
    )
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(workspace.polygon_uv, dtype=np.int32)
    cv2.fillPoly(roi_mask, [polygon], 255)
    valid_depth = np.isfinite(z) & (z > 0.0)
    foreground = (
        valid_depth
        & (plane_height >= float(config.proposal_min_height_m))
        & (plane_height <= float(config.proposal_max_height_m))
        & (roi_mask > 0)
    ).astype(np.uint8) * 255
    kernel_size = max(3, int(config.morphology_kernel_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    warnings: list[str] = []
    inner = cv2.erode(roi_mask, np.ones((5, 5), dtype=np.uint8))
    boundary = cv2.subtract(roi_mask, inner)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.proposal_min_area_px:
            continue
        if area > config.proposal_max_area_px:
            warnings.append("检测到面积过大的连通区域，可能存在物体粘连")
            continue
        contour_mask = np.zeros_like(roi_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        if cv2.countNonZero(cv2.bitwise_and(contour_mask, boundary)) > 0:
            warnings.append("有候选物体接触工作区边界；保留标注框，但禁止推荐抓取")
        x0, y0, box_width, box_height = cv2.boundingRect(contour)
        boxes.append([x0, y0, box_width, box_height])
    boxes.sort(key=lambda box: (box[1], box[0]))
    return {
        "image_size": [width, height],
        "boxes": boxes,
        "roi_polygon_uv": workspace.polygon_uv,
        "foreground_fraction": float(cv2.countNonZero(foreground) / foreground.size),
        "warnings": list(dict.fromkeys(warnings)),
        "thresholds": {
            "min_height_m": config.proposal_min_height_m,
            "max_height_m": config.proposal_max_height_m,
            "min_area_px": config.proposal_min_area_px,
            "max_area_px": config.proposal_max_area_px,
        },
    }


@dataclass(frozen=True)
class VisionDetection:
    detection_id: str
    class_name: str
    kind: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    area_px: float
    depth_m: float | None
    depth_spread_m: float | None
    depth_valid_fraction: float
    graspable: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TabletopVisionDetector:
    """Detect the first-version tabletop objects with color, contour and depth cues."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        roi_polygon_uv: list[list[int]] | None = None,
        depth_proposal_boxes: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("color frame must be a BGR image")
        if depth_raw.shape[:2] != color_bgr.shape[:2]:
            raise ValueError("aligned color and depth frame sizes do not match")

        height, width = color_bgr.shape[:2]
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        polygon_uv: list[list[int]] | None = None
        if roi_polygon_uv:
            polygon = np.asarray(roi_polygon_uv, dtype=np.int32)
            if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
                raise ValueError("workspace image polygon is invalid")
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            cv2.fillPoly(roi_mask, [polygon], 255)
            x, y, roi_width, roi_height = cv2.boundingRect(polygon)
            x1, y1, x2, y2 = x, y, x + roi_width, y + roi_height
            polygon_uv = polygon.tolist()
        else:
            x1, y1, x2, y2 = self._clamped_roi(width, height)
            roi_mask[y1:y2, x1:x2] = 255

        green = cv2.inRange(
            hsv,
            np.asarray(self.config.green_hsv_low, dtype=np.uint8),
            np.asarray(self.config.green_hsv_high, dtype=np.uint8),
        )
        gray_range = cv2.inRange(
            hsv,
            np.asarray(self.config.gray_hsv_low, dtype=np.uint8),
            np.asarray(self.config.gray_hsv_high, dtype=np.uint8),
        )
        green = self._clean_mask(cv2.bitwise_and(green, roi_mask))
        adaptive_size = max(3, int(self.config.gray_adaptive_block_px))
        if adaptive_size % 2 == 0:
            adaptive_size += 1
        gray_contrast = cv2.adaptiveThreshold(
            hsv[:, :, 2],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_size,
            float(self.config.gray_adaptive_c),
        )
        gray = self._clean_mask(
            cv2.bitwise_and(cv2.bitwise_and(gray_range, gray_contrast), roi_mask)
        )

        detections: list[VisionDetection] = []
        detections.extend(
            self._green_detections(hsv, green, depth_raw, depth_scale, (x1, y1, x2, y2), roi_mask)
        )
        if depth_proposal_boxes is None:
            detections.extend(
                self._gray_detections(gray, depth_raw, depth_scale, (x1, y1, x2, y2), roi_mask)
            )
        else:
            detections.extend(
                self._gray_depth_detections(
                    hsv,
                    gray,
                    depth_raw,
                    depth_scale,
                    depth_proposal_boxes,
                    (x1, y1, x2, y2),
                    roi_mask,
                )
            )
        detections.sort(
            key=lambda item: (item.kind != "destination", -item.confidence, -item.area_px)
        )
        detections = [
            VisionDetection(
                detection_id=f"vision-{index + 1}",
                class_name=item.class_name,
                kind=item.kind,
                confidence=item.confidence,
                bbox_xywh=item.bbox_xywh,
                center_uv=item.center_uv,
                area_px=item.area_px,
                depth_m=item.depth_m,
                depth_spread_m=item.depth_spread_m,
                depth_valid_fraction=item.depth_valid_fraction,
                graspable=item.graspable,
                rejection_reason=item.rejection_reason,
            )
            for index, item in enumerate(detections)
        ]
        recommended = self.recommend_object(detections)
        destination = self.recommend_destination(detections)
        warnings: list[str] = []
        if not recommended:
            warnings.append("没有检测到深度有效且轮廓完整的可抓取物体")
        if not destination:
            warnings.append("没有检测到深度有效的绿色实心托盘")
        return {
            "roi_xyxy": [x1, y1, x2, y2],
            "roi_polygon_uv": polygon_uv,
            "detections": [item.to_dict() for item in detections],
            "recommended_object_id": recommended.detection_id if recommended else None,
            "destination_id": destination.detection_id if destination else None,
            "warnings": warnings,
        }

    @staticmethod
    def recommend_object(
        detections: list[VisionDetection], target_class: str | None = None
    ) -> VisionDetection | None:
        candidates = [
            item
            for item in detections
            if item.kind == "object"
            and item.graspable
            and (target_class is None or item.class_name == target_class)
        ]
        return max(
            candidates, key=lambda item: (item.confidence, item.depth_valid_fraction), default=None
        )

    @staticmethod
    def recommend_destination(detections: list[VisionDetection]) -> VisionDetection | None:
        candidates = [
            item
            for item in detections
            if item.kind == "destination"
            and item.depth_m is not None
            and item.rejection_reason is None
        ]
        return max(candidates, key=lambda item: (item.confidence, item.area_px), default=None)

    def _green_detections(
        self,
        hsv: np.ndarray,
        mask: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        roi: tuple[int, int, int, int],
        roi_mask: np.ndarray,
    ) -> list[VisionDetection]:
        result: list[VisionDetection] = []
        for contour in self._contours(mask):
            area = float(cv2.contourArea(contour))
            if area < self.config.object_min_area_px:
                continue
            geometry = self._geometry(contour)
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            saturation_values = hsv[:, :, 1][contour_mask > 0]
            median_saturation = (
                float(np.median(saturation_values)) if saturation_values.size else 0.0
            )
            rotated_aspect = float(geometry["rotated_aspect_ratio"])
            tray_shape = (
                area >= self.config.tray_min_area_px
                and median_saturation >= self.config.tray_min_saturation
                and self.config.tray_min_rotated_aspect_ratio
                <= rotated_aspect
                <= self.config.tray_max_rotated_aspect_ratio
                and 4 <= geometry["vertices"] <= self.config.tray_max_vertices
            )
            if tray_shape:
                aspect_center = (
                    self.config.tray_min_rotated_aspect_ratio
                    + self.config.tray_max_rotated_aspect_ratio
                ) / 2.0
                aspect_half_range = max(
                    0.01,
                    (
                        self.config.tray_max_rotated_aspect_ratio
                        - self.config.tray_min_rotated_aspect_ratio
                    )
                    / 2.0,
                )
                aspect_score = max(
                    0.0, 1.0 - abs(rotated_aspect - aspect_center) / aspect_half_range
                )
                saturation_score = min(1.0, median_saturation / 180.0)
                result.append(
                    self._make_detection(
                        contour,
                        mask,
                        depth_raw,
                        depth_scale,
                        roi,
                        roi_mask,
                        class_name="green_tray",
                        kind="destination",
                        shape_score=min(
                            1.0,
                            0.45 * geometry["rotated_rectangularity"]
                            + 0.30 * aspect_score
                            + 0.25 * saturation_score,
                        ),
                        depth_max_spread_m=self.config.tray_depth_max_spread_m,
                    )
                )
                continue
            if area > self.config.object_max_area_px:
                continue
            circularity = geometry["circularity"]
            if circularity < self.config.cylinder_min_circularity:
                continue
            result.append(
                self._make_detection(
                    contour,
                    mask,
                    depth_raw,
                    depth_scale,
                    roi,
                    roi_mask,
                    class_name="green_cylinder",
                    kind="object",
                    shape_score=min(1.0, circularity / 0.9),
                )
            )
        return result

    def _gray_detections(
        self,
        mask: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        roi: tuple[int, int, int, int],
        roi_mask: np.ndarray,
    ) -> list[VisionDetection]:
        result: list[VisionDetection] = []
        for contour in self._contours(mask):
            area = float(cv2.contourArea(contour))
            minimum_area = min(self.config.object_min_area_px, self.config.proposal_min_area_px)
            if not minimum_area <= area <= self.config.object_max_area_px:
                continue
            geometry = self._geometry(contour)
            aspect_score = max(
                0.0,
                1.0
                - abs(1.0 - geometry["aspect_ratio"]) / self.config.depth_square_aspect_tolerance,
            )
            if aspect_score <= 0.0 or not 4 <= geometry["vertices"] <= 8:
                continue
            result.append(
                self._make_detection(
                    contour,
                    mask,
                    depth_raw,
                    depth_scale,
                    roi,
                    roi_mask,
                    class_name="gray_cube",
                    kind="object",
                    shape_score=min(1.0, 0.5 * aspect_score + 0.5 * geometry["rectangularity"]),
                )
            )
        return result

    def _gray_depth_detections(
        self,
        hsv: np.ndarray,
        gray_mask: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        proposal_boxes: list[list[int]],
        roi: tuple[int, int, int, int],
        roi_mask: np.ndarray,
    ) -> list[VisionDetection]:
        """Classify depth-isolated candidates without requiring a global gray contour."""
        result: list[VisionDetection] = []
        image_height, image_width = hsv.shape[:2]
        low = np.asarray(self.config.gray_hsv_low, dtype=np.uint8)
        high = np.asarray(self.config.gray_hsv_high, dtype=np.uint8)
        for raw_box in proposal_boxes:
            if len(raw_box) != 4:
                continue
            x, y, width, height = (int(value) for value in raw_box)
            x = max(0, min(image_width - 1, x))
            y = max(0, min(image_height - 1, y))
            width = max(1, min(image_width - x, width))
            height = max(1, min(image_height - y, height))
            area = float(width * height)
            if not self.config.object_min_area_px <= area <= self.config.object_max_area_px:
                continue
            crop = hsv[y : y + height, x : x + width]
            gray_fraction = float(np.mean(np.all((crop >= low) & (crop <= high), axis=2)))
            if gray_fraction < self.config.gray_depth_min_fraction:
                continue
            aspect_ratio = width / float(height)
            aspect_score = max(
                0.0,
                1.0 - abs(1.0 - aspect_ratio) / self.config.depth_square_aspect_tolerance,
            )
            if aspect_score <= 0.0:
                continue
            contour = np.asarray(
                [
                    [[x, y]],
                    [[x + width - 1, y]],
                    [[x + width - 1, y + height - 1]],
                    [[x, y + height - 1]],
                ],
                dtype=np.int32,
            )
            result.append(
                self._make_detection(
                    contour,
                    gray_mask,
                    depth_raw,
                    depth_scale,
                    roi,
                    roi_mask,
                    class_name="gray_cube",
                    kind="object",
                    shape_score=min(1.0, 0.65 * aspect_score + 0.35 * gray_fraction),
                )
            )
        return result

    def _make_detection(
        self,
        contour: np.ndarray,
        color_mask: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        roi: tuple[int, int, int, int],
        roi_mask: np.ndarray,
        *,
        class_name: str,
        kind: str,
        shape_score: float,
        depth_max_spread_m: float | None = None,
    ) -> VisionDetection:
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        center = (
            (round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"]))
            if moments["m00"]
            else (x + width // 2, y + height // 2)
        )
        contour_mask = np.zeros_like(color_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        depth_m, spread_m, valid_fraction = self._depth_stats(depth_raw, depth_scale, contour_mask)
        geometry = self._geometry(contour)
        touching_roi = self._touches_roi((x, y, width, height), roi, contour_mask, roi_mask)
        rejection_reason: str | None = None
        if geometry["solidity"] < self.config.contour_min_solidity:
            rejection_reason = "轮廓不完整或疑似粘连"
        elif touching_roi:
            rejection_reason = "目标接触视觉区域边界"
        elif depth_m is None:
            rejection_reason = "目标区域没有足够的有效深度"
        elif spread_m is not None and spread_m > (
            self.config.depth_max_spread_m if depth_max_spread_m is None else depth_max_spread_m
        ):
            rejection_reason = "目标区域深度变化过大，疑似遮挡或粘连"
        depth_score = min(1.0, valid_fraction / 0.8)
        confidence = float(
            np.clip(0.48 * shape_score + 0.27 * geometry["solidity"] + 0.25 * depth_score, 0.0, 1.0)
        )
        return VisionDetection(
            detection_id="",
            class_name=class_name,
            kind=kind,
            confidence=confidence,
            bbox_xywh=(x, y, width, height),
            center_uv=center,
            area_px=float(cv2.contourArea(contour)),
            depth_m=depth_m,
            depth_spread_m=spread_m,
            depth_valid_fraction=valid_fraction,
            graspable=kind == "object" and rejection_reason is None,
            rejection_reason=rejection_reason,
        )

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        size = max(1, int(self.config.morphology_kernel_px))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _contours(mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    @staticmethod
    def _geometry(contour: np.ndarray) -> dict[str, float | int]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        _, _, width, height = cv2.boundingRect(contour)
        (_, _), (rotated_width, rotated_height), _ = cv2.minAreaRect(contour)
        rotated_short = min(rotated_width, rotated_height)
        rotated_long = max(rotated_width, rotated_height)
        rotated_area = rotated_width * rotated_height
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True) if perimeter else contour
        return {
            "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0,
            "solidity": area / hull_area if hull_area else 0.0,
            "rectangularity": area / float(width * height) if width and height else 0.0,
            "aspect_ratio": width / float(height) if height else 0.0,
            "rotated_aspect_ratio": (rotated_long / rotated_short if rotated_short > 0.0 else 0.0),
            "rotated_rectangularity": area / rotated_area if rotated_area > 0.0 else 0.0,
            "vertices": len(approx),
        }

    @staticmethod
    def _depth_stats(
        depth_raw: np.ndarray, depth_scale: float, mask: np.ndarray
    ) -> tuple[float | None, float | None, float]:
        selected = depth_raw[mask > 0].astype(np.float64) * float(depth_scale)
        valid = selected[np.isfinite(selected) & (selected > 0.0)]
        fraction = float(valid.size / selected.size) if selected.size else 0.0
        if valid.size < 9 or fraction < 0.35:
            return None, None, fraction
        lower, median, upper = np.percentile(valid, [10, 50, 90])
        return float(median), float(upper - lower), fraction

    def _clamped_roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        raw_x1, raw_y1, raw_x2, raw_y2 = self.config.roi_xyxy
        x1 = max(0, min(width - 1, int(raw_x1)))
        y1 = max(0, min(height - 1, int(raw_y1)))
        x2 = max(x1 + 1, min(width, int(raw_x2)))
        y2 = max(y1 + 1, min(height, int(raw_y2)))
        return x1, y1, x2, y2

    @staticmethod
    def _touches_roi(
        bbox: tuple[int, int, int, int],
        roi: tuple[int, int, int, int],
        contour_mask: np.ndarray,
        roi_mask: np.ndarray,
    ) -> bool:
        x, y, width, height = bbox
        x1, y1, x2, y2 = roi
        margin = 2
        touches_bounds = (
            x <= x1 + margin
            or y <= y1 + margin
            or x + width >= x2 - margin
            or y + height >= y2 - margin
        )
        kernel = np.ones((5, 5), dtype=np.uint8)
        inner = cv2.erode(roi_mask, kernel)
        inner_boundary = cv2.subtract(roi_mask, inner)
        touches_polygon = cv2.countNonZero(cv2.bitwise_and(contour_mask, inner_boundary)) > 0
        return touches_bounds or touches_polygon


def build_auto_annotations(
    detection_result: dict[str, Any],
    target_class: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Validate a single-class scene and convert accepted boxes to YOLO coordinates."""
    if target_class not in AUTO_CAPTURE_CLASSES:
        raise ValueError("unsupported automatic annotation class")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    detections = list(detection_result.get("detections", []))
    reasons: list[str] = []
    annotations: list[dict[str, Any]] = []
    if target_class == "empty_table":
        if detections:
            reasons.append("空桌面负样本中检测到了物体")
    else:
        matching = [item for item in detections if item.get("class_name") == target_class]
        other_classes = sorted(
            {
                str(item.get("class_name"))
                for item in detections
                if item.get("class_name") != target_class
            }
        )
        if other_classes:
            reasons.append(f"工作区包含其他类别：{', '.join(other_classes)}")
        if not matching:
            reasons.append("没有检测到所选类别的物体")
        rejected = [item for item in matching if item.get("rejection_reason")]
        if rejected:
            details = sorted({str(item["rejection_reason"]) for item in rejected})
            reasons.append(f"存在不可用候选：{'; '.join(details)}")
        if not reasons:
            class_id = YOLO_CLASSES.index(target_class)
            for item in matching:
                x, y, width, height = (int(value) for value in item["bbox_xywh"])
                annotations.append(
                    {
                        "class_id": class_id,
                        "class_name": target_class,
                        "bbox_xywh": [x, y, width, height],
                        "yolo_xywh": [
                            (x + width / 2.0) / image_width,
                            (y + height / 2.0) / image_height,
                            width / image_width,
                            height / image_height,
                        ],
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                )
    return {
        "valid": not reasons,
        "target_class": target_class,
        "image_size": [image_width, image_height],
        "annotations": annotations,
        "reasons": reasons,
        "roi_xyxy": detection_result.get("roi_xyxy"),
        "roi_polygon_uv": detection_result.get("roi_polygon_uv"),
    }
