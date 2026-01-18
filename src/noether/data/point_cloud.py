#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Contains point cloud utilities."""

import numpy as np


class PointCloudRegistration:
    """Class with the interface to register point clouds.

    - The point clouds are registered by estimating the center shift and scaling of the source point cloud to the target point cloud.
    - The source point cloud is transformed to the target point cloud using the estimated center shift and scaling.
    - A center_shift or xyz_scale can be provided to transform the source point cloud to the target point cloud.

    """

    def __init__(self, center_shift: np.ndarray | None = None, xyz_scale: float | None = None):
        """Initialize the PointCloudRegistration class.

        Args:
            center_shift: The center shift of the point cloud. Defaults to None.
            xyz_scale: The scaling of the point cloud. Defaults to None.
        """
        self.center_shift = center_shift
        self.xyz_scale = xyz_scale

    def estimate_transforms(self, src_points: np.ndarray, target_points: np.ndarray) -> None:
        """Estimate the transformation parameters for registering the source point cloud to the target point cloud. Only 3D point clouds are supported.

        Args:
            src_points: The source point cloud. Only 3D point clouds are supported.
            target_points: The target point cloud. Only 3D point clouds are supported.
        """

        assert src_points.shape[1] == 3 and target_points.shape[1] == 3

        src_points = src_points.copy()

        # normalize span
        src_span = np.max(src_points, axis=0) - np.min(src_points, axis=0)
        target_span = np.max(target_points, axis=0) - np.min(target_points, axis=0)
        xyz_scale = (target_span / src_span).mean()
        src_points *= xyz_scale

        # estimate center shift
        center_shift = np.zeros((1, 3))

        src_mean_0 = (np.max(src_points[:, 0]) + np.min(src_points[:, 0])) / 2
        target_mean_0 = (np.max(target_points[:, 0]) + np.min(target_points[:, 0])) / 2
        center_shift[0, 0] = target_mean_0 - src_mean_0

        src_mean_1 = (np.max(src_points[:, 1]) + np.min(src_points[:, 1])) / 2
        target_mean_1 = (np.max(target_points[:, 1]) + np.min(target_points[:, 1])) / 2
        center_shift[0, 1] = target_mean_1 - src_mean_1

        center_shift[0, 2] = np.min(target_points[:, 2]) - np.min(src_points[:, 2])

        self.center_shift = center_shift.astype(np.float32)
        self.xyz_scale = xyz_scale

    def transform(self, src_points: np.ndarray) -> np.ndarray:
        """Transform the source point cloud to the target point cloud.

        Args:
            src_points: The source point cloud.

        Returns:
            The transformed source point.
        """
        assert self.center_shift is not None and self.xyz_scale is not None

        registered_src_points = src_points * float(self.xyz_scale) + self.center_shift
        return registered_src_points  # type: ignore

    def __repr__(self) -> str:
        return f"PointCloudRegistration(center_shift={self.center_shift}, xyz_scale={self.xyz_scale})"
