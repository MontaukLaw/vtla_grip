from __future__ import annotations

import numpy as np


def transform_point(transform_4x4: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    """Apply a homogeneous rigid transform to one XYZ point."""
    transform = np.asarray(transform_4x4, dtype=np.float64)
    point = np.asarray(point_xyz, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if point.shape != (3,):
        raise ValueError("point_xyz must have shape (3,)")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("invalid homogeneous transform last row")
    result = transform @ np.append(point, 1.0)
    return result[:3]
