from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


@dataclass(frozen=True)
class CalibrationData:
    path: Path
    camera_mm: np.ndarray
    base_mm: np.ndarray
    sample_indices: list[int]
    tool_orientations_deg: np.ndarray


def latest_calibration_record(directory: str | Path = "records/calibration") -> Path:
    files = sorted(Path(directory).glob("blue-dot-*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no blue-dot calibration records found in {directory}")
    return files[-1]


def load_calibration_data(path: str | Path) -> CalibrationData:
    source = Path(path)
    camera: list[list[float]] = []
    base: list[list[float]] = []
    indices: list[int] = []
    orientations: list[list[float]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "sample":
            continue
        camera_xyz = event.get("camera_xyz_m")
        tcp_pose = event.get("robot_tcp_pose_mm_deg")
        if not isinstance(camera_xyz, list) or len(camera_xyz) != 3:
            raise ValueError(f"line {line_number}: invalid camera_xyz_m")
        if not isinstance(tcp_pose, list) or len(tcp_pose) != 6:
            raise ValueError(f"line {line_number}: invalid robot_tcp_pose_mm_deg")
        values = [float(value) for value in [*camera_xyz, *tcp_pose[:3]]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {line_number}: non-finite calibration value")
        camera.append([value * 1000.0 for value in values[:3]])
        base.append(values[3:])
        orientations.append([float(value) for value in tcp_pose[3:]])
        indices.append(int(event.get("sample_index", len(indices) + 1)))
    if len(camera) < 6:
        raise ValueError("at least 6 calibration samples are required")
    return CalibrationData(
        source,
        np.asarray(camera),
        np.asarray(base),
        indices,
        np.asarray(orientations),
    )


def solve_rigid_transform(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target points must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least 3 point pairs are required")
    if np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-6) < 2:
        raise ValueError("calibration camera points are collinear or degenerate")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def residuals_mm(
    transform: np.ndarray, source_xyz: np.ndarray, target_xyz: np.ndarray
) -> np.ndarray:
    predicted = (transform[:3, :3] @ np.asarray(source_xyz).T).T + transform[:3, 3]
    return np.linalg.norm(predicted - np.asarray(target_xyz), axis=1)


def ransac_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    threshold_mm: float = 15.0,
    iterations: int = 500,
    seed: int = 20260827,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if len(source) < 4:
        return solve_rigid_transform(source, target), np.ones(len(source), dtype=bool)
    generator = random.Random(seed)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for _ in range(iterations):
        chosen = generator.sample(range(len(source)), 3)
        try:
            candidate = solve_rigid_transform(source[chosen], target[chosen])
        except ValueError:
            continue
        errors = residuals_mm(candidate, source, target)
        mask = errors <= threshold_mm
        if mask.sum() < 3:
            continue
        score = (int(mask.sum()), -float(np.median(errors[mask])))
        if best_score is None or score > best_score:
            best_score, best_mask = score, mask
    if best_mask is None or int(best_mask.sum()) < max(3, math.ceil(len(source) * 0.6)):
        raise ValueError("RANSAC could not find a stable transform; check blue-dot samples")
    return solve_rigid_transform(source[best_mask], target[best_mask]), best_mask


def _metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": len(values),
        "mean_mm": float(values.mean()),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "max_mm": float(values.max()),
        "median_mm": float(np.median(values)),
    }


def solve_calibration(
    source_path: str | Path,
    *,
    threshold_mm: float = 15.0,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    data = load_calibration_data(source_path)
    count = len(data.camera_mm)
    validation_step = max(3, round(1.0 / max(0.1, min(validation_fraction, 0.4))))
    validation_mask = np.asarray([(index + 1) % validation_step == 0 for index in range(count)])
    if validation_mask.sum() < 2:
        validation_mask[-2:] = True
    train_mask = ~validation_mask
    validation_transform, train_inliers = ransac_transform(
        data.camera_mm[train_mask], data.base_mm[train_mask], threshold_mm=threshold_mm
    )
    validation_errors = residuals_mm(
        validation_transform, data.camera_mm[validation_mask], data.base_mm[validation_mask]
    )
    preliminary_errors = residuals_mm(validation_transform, data.camera_mm, data.base_mm)
    final_inliers = preliminary_errors <= threshold_mm
    if int(final_inliers.sum()) < max(6, math.ceil(count * 0.6)):
        raise ValueError("too few inliers remain after validation")
    final_transform = solve_rigid_transform(
        data.camera_mm[final_inliers], data.base_mm[final_inliers]
    )
    final_errors = residuals_mm(final_transform, data.camera_mm, data.base_mm)
    accepted = final_errors <= threshold_mm
    rotation = final_transform[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "source_record": str(data.path.resolve()),
        "method": "Kabsch rigid transform with deterministic RANSAC and interleaved holdout",
        "units": {"transform_translation": "mm", "input_camera": "m", "input_base": "mm"},
        "transform_name": "T_base_camera",
        "transform_4x4": final_transform.tolist(),
        "fixed_tool_orientation_deg": np.median(data.tool_orientations_deg, axis=0).tolist(),
        "tool_orientation_range_deg": np.ptp(data.tool_orientations_deg, axis=0).tolist(),
        "sample_count": count,
        "inlier_count": int(accepted.sum()),
        "outlier_sample_indices": [
            data.sample_indices[index] for index, is_inlier in enumerate(accepted) if not is_inlier
        ],
        "training": {
            **_metrics(preliminary_errors[train_mask]),
            "ransac_inliers": int(train_inliers.sum()),
        },
        "validation": _metrics(validation_errors),
        "final_all_samples": _metrics(final_errors),
        "per_sample": [
            {
                "sample_index": data.sample_indices[index],
                "residual_mm": float(final_errors[index]),
                "accepted": bool(accepted[index]),
            }
            for index in range(count)
        ],
        "quality": {
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_orthogonality_error": orthogonality_error,
            "hover_ready": bool(
                _metrics(validation_errors)["rms_mm"] <= 10.0
                and _metrics(validation_errors)["max_mm"] <= 15.0
            ),
        },
        "warning": (
            "Blue-dot-to-TCP offset is not independently solved. This transform is only suitable "
            "for the validated, nearly fixed tool orientation until that offset is measured."
        ),
    }
    return report


def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Solve D435 camera-to-RM65 Base calibration")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(config.calibration.output_path))
    parser.add_argument(
        "--threshold-mm", type=float, default=config.calibration.ransac_threshold_mm
    )
    args = parser.parse_args()
    source = args.input or latest_calibration_record()
    report = solve_calibration(source, threshold_mm=args.threshold_mm)
    path = save_report(report, args.output)
    print(json.dumps({"output": str(path.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
