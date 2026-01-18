#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.layers import LayerScaleConfig


class LayerScale(nn.Module):
    """LayerScale module scales the input tensor by a learnable parameter gamma."""

    def __init__(self, config: LayerScaleConfig):
        """
        Initialize the LayerScale module.
        Args:
            hidden_dim: Number of dimensions of the input tensor to be scaled.
            init_scale: Initial gamme scale value. Defaults to 1e-5.
        """

        super().__init__()
        if config.init_values is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.full(size=(config.hidden_dim,), fill_value=config.init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the LayerScale module.

        Args:
            x: Input tensor to be scaled.

        Returns:
            Tensor scaled by the gamma parameter.
        """

        if self.gamma is None:
            return x
        return x * self.gamma
