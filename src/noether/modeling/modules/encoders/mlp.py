#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from torch import nn

from noether.core.schemas.modules.layers import LinearProjectionConfig
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.layers import LinearProjection


class MlpEncoder(nn.Module):
    """A 2-layer MLP encoder. Can be use to encode (i.e., embed) the input data."""

    def __init__(self, input_dim: int, hidden_dim: int, init_weights="truncnormal002"):
        """
        Initialize the MlpEncoder.

        Args:
            input_dim: Number of dimensions of the input tensor.
            hidden_dim: Hidden dimensionality of the network.
            init_weights: Initialization method for the weight matrixes of the linear layers. Defaults to "truncnormal002".
        """
        super().__init__()

        assert input_dim > 0, "Input dimension must be positive"
        assert hidden_dim > 0, "Hidden dimension must be positive"

        config1 = LinearProjectionConfig(
            input_dim=input_dim,
            output_dim=hidden_dim,
            init_weights=init_weights,
        )  # type: ignore[call-arg]
        config2 = LinearProjectionConfig(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            init_weights=init_weights,
        )  # type: ignore[call-arg]
        self.layer = nn.Sequential(
            LinearProjection(config1),
            Activation.GELU.value,
            LinearProjection(config2),
        )

    def forward(self, x):
        """Forward method of the MlpEncoder.

        Args:
            x: Tensor of input data.

        Returns:
            Embedded/encoded tensor.
        """

        return self.layer(x)
