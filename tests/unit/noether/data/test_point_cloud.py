#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np
import pytest

from noether.data.point_cloud import PointCloudRegistration


def test_init():
    """Test the initialization of the PointCloudRegistration class."""
    registration = PointCloudRegistration()
    assert registration.center_shift is None
    assert registration.xyz_scale is None

    registration = PointCloudRegistration(center_shift=np.array([[1.0, 2.0, 3.0]], dtype=np.float32), xyz_scale=1.5)
    assert registration.center_shift.shape == (1, 3)
    assert registration.center_shift[0, 0] == 1.0
    assert registration.center_shift[0, 1] == 2.0
    assert registration.center_shift[0, 2] == 3.0
    assert registration.xyz_scale == 1.5


def test_estimate_transforms():
    """Test the estimate_transforms method of the PointCloudRegistration class."""
    src_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    target_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)

    registration = PointCloudRegistration()
    registration.estimate_transforms(src_points, target_points)

    assert registration.xyz_scale == pytest.approx(1.0, rel=1e-3)
    assert registration.center_shift.shape == (1, 3)
    assert registration.center_shift[0, 0] == pytest.approx(1.0, rel=1e-3)
    assert registration.center_shift[0, 1] == pytest.approx(1.0, rel=1e-3)
    assert registration.center_shift[0, 2] == pytest.approx(1.0, rel=1e-3)

    target_points = np.array([[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.float32)
    with pytest.raises(AssertionError):
        registration.estimate_transforms(src_points, target_points)
        registration.estimate_transforms(target_points, src_points)


def test_transform():
    """Test the transform method of the PointCloudRegistration class."""
    src_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    target_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)

    registration = PointCloudRegistration()
    registration.estimate_transforms(src_points, target_points)
    assert registration.center_shift is not None and registration.xyz_scale is not None
    transformed_points = registration.transform(src_points)

    expected_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    np.testing.assert_allclose(transformed_points, expected_points, rtol=1e-3)
    assert transformed_points.dtype == np.float32


def test_transform_without_shift_and_scale():
    """Test if assertion error is raised when transform is called without center_shift and xyz_scale."""
    src_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

    registration = PointCloudRegistration()
    with pytest.raises(AssertionError):
        registration.transform(src_points)

    registration = PointCloudRegistration(xyz_scale=1.5)
    with pytest.raises(AssertionError):
        registration.transform(src_points)

    registration = PointCloudRegistration(center_shift=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    with pytest.raises(AssertionError):
        registration.transform(src_points)


def test_repr():
    """Test the __repr__ method of the PointCloudRegistration class."""
    registration = PointCloudRegistration(center_shift=np.array([[1.0, 2.0, 3.0]], dtype=np.float32), xyz_scale=1.5)
    repr_str = repr(registration)
    assert repr_str == "PointCloudRegistration(center_shift=[[1. 2. 3.]], xyz_scale=1.5)"
