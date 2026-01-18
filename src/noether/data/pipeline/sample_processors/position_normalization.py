#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class PositionNormalizationSampleProcessor(SampleProcessor):
    """Pre-processes data on a sample-level to normalize positions."""

    def __init__(
        self,
        items: set[str],
        raw_pos_min: Sequence[float],
        raw_pos_max: Sequence[float],
        scale: int | float = 1000,
    ):
        """

        Args:
            items: The position items to normalize.
            raw_pos_min: The minimum position in the source domain.
            raw_pos_max: The maximum position in the source domain.
            scale: The maximum value of the position. Defaults to 1000.
        """
        assert len(raw_pos_min) == len(raw_pos_max), "Raw position min and max must have the same length."

        self.items = items
        self.scale = scale
        self.raw_pos_min_tensor = torch.tensor(raw_pos_min)
        self.raw_pos_max_tensor = torch.tensor(raw_pos_max)
        self.raw_size = self.raw_pos_max_tensor - self.raw_pos_min_tensor

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-processes data on a sample-level to normalize positions.

        Args:
            input_sample: Dictionary of a single sample.

        Return:
           Preprocessed copy of `input_sample` with positions normalized.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        # process
        for item in self.items:
            # coordinate dimension is the last dimension of the tensor, this allows normalizing arbitrarily
            # shaped tensors (e.g., 3D positions can be contained in tensors of shape [24, 16, 3], [16, 3])
            # https://pytorch.org/docs/stable/notes/broadcasting.html
            output_sample[item] = (output_sample[item] - self.raw_pos_min_tensor).div_(self.raw_size).mul_(self.scale)

        return output_sample

    def inverse(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the normalization from the __call__ method of a single item in the batch.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The same name and the denormalized value.
        """
        if key not in self.items:
            return key, value
        # shapes are broadcasted (https://pytorch.org/docs/stable/notes/broadcasting.html)
        raw_pos_min = self.raw_pos_min_tensor.to(value.device)
        raw_size = self.raw_size.to(value.device)
        denormalized_value = (value / self.scale).mul_(raw_size).add_(raw_pos_min)
        return key, denormalized_value
