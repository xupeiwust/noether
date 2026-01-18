#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline import SampleProcessor


class SurfaceMaskSampleProcessor(SampleProcessor):
    """"""

    def __init__(
        self,
        item: str,
        num_surface_points: int,
        num_volume_points: int,
    ):
        """_summary_

        Args:
            num_surface_points: _description_
            num_volume_points: _description_
        """
        self.item = item
        self.num_surface_points = num_surface_points
        self.num_volume_points = num_volume_points

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """_summary_

        Args:
            input_sample: _description_

        Returns:
            _description_
        """
        output_sample = self.save_copy(input_sample)

        surface_mask = torch.zeros(self.num_surface_points + self.num_volume_points)
        surface_mask[: self.num_surface_points] = 1.0
        output_sample[self.item] = surface_mask.bool()

        return output_sample
