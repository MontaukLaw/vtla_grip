from __future__ import annotations

import numpy as np


def median_depth_m(
    depth_raw: np.ndarray,
    u: int,
    v: int,
    depth_scale: float,
    radius: int = 2,
    min_depth_m: float = 0.10,
    max_depth_m: float = 1.50,
) -> float | None:
    """Return robust local depth in metres, ignoring zero and out-of-range pixels."""
    if depth_raw.ndim != 2:
        raise ValueError("depth_raw must be a two-dimensional image")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    height, width = depth_raw.shape
    if not (0 <= u < width and 0 <= v < height):
        return None

    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    values_m = depth_raw[y0:y1, x0:x1].astype(np.float64) * depth_scale
    valid = values_m[(values_m > 0) & (values_m >= min_depth_m) & (values_m <= max_depth_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
