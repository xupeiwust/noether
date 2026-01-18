#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline import SampleProcessor


class DefaultTensorSampleProcessor(SampleProcessor):
    """Create a tensor with a fixed dummy value, with a specified size."""

    def __init__(
        self,
        item_key_name: str,
        feature_dim: int,
        size: int = None,
        matching_item_key: str = None,
        default_value: float = 0.0,
    ):
        """

        Args:
            item_key_name: _description_
            default_value: _description_
            feature_dim: _description_
            size: _description_
            matching_item_key: _description_
        """
        assert size is not None or matching_item_key is not None, (
            "feature_dim or matching_item_key must be specified. Otherwise size cannot be determined."
        )
        assert item_key_name is not None, "key_name must be specified."

        self.item_key_name = item_key_name
        self.feature_dim = feature_dim
        self.size = size
        self.matching_item_key = matching_item_key
        self.default_value = default_value

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """_summary_

        Args:
            input_sample: _description_

        Returns:
            _description_
        """
        # copy to avoid changing method input

        output_sample = self.save_copy(input_sample)

        dim = self.size or output_sample[self.matching_item_key].shape[0]
        output_sample[self.item_key_name] = torch.empty(dim, self.feature_dim).fill_(self.default_value)

        return output_sample
