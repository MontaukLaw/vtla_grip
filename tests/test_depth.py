import numpy as np

from vtla_grip.depth import median_depth_m


def test_median_depth_ignores_zero_and_outliers() -> None:
    depth = np.array(
        [
            [0, 500, 501],
            [499, 5000, 502],
            [498, 500, 0],
        ],
        dtype=np.uint16,
    )
    result = median_depth_m(depth, 1, 1, 0.001, radius=1, max_depth_m=1.5)
    assert result == 0.5


def test_median_depth_returns_none_outside_image() -> None:
    depth = np.ones((3, 3), dtype=np.uint16) * 500
    assert median_depth_m(depth, -1, 0, 0.001) is None


def test_median_depth_returns_none_without_valid_samples() -> None:
    depth = np.zeros((3, 3), dtype=np.uint16)
    assert median_depth_m(depth, 1, 1, 0.001) is None
