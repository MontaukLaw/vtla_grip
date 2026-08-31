import numpy as np
import pytest

from vtla_grip.transforms import transform_point


def test_transform_point_applies_rotation_and_translation() -> None:
    transform = np.array(
        [
            [0, -1, 0, 100],
            [1, 0, 0, 200],
            [0, 0, 1, 300],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    result = transform_point(transform, np.array([10, 20, 30], dtype=float))
    np.testing.assert_allclose(result, [80, 210, 330])


def test_transform_point_rejects_bad_transform() -> None:
    with pytest.raises(ValueError):
        transform_point(np.eye(3), np.zeros(3))
