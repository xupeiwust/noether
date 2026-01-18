#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn as nn

from noether.core.schemas.models import TransformerConfig
from noether.core.schemas.modules import TransformerBlockConfig
from noether.modeling.modules.blocks import TransformerBlock


class Transformer(nn.Module):
    """Implementation of a Transformer model."""

    def __init__(
        self,
        config: TransformerConfig,
    ):
        """
        Args:
            config: Configuration of the Transformer model.
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=TransformerBlockConfig(
                        **config.model_dump(),  # pass down all relevant config parameters, everything that is not used is ignored by the TransformerBlockConfig
                    ),
                )
                for _ in range(config.depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
            attn_kwargs: Additional arguments for the attention mechanism.
        Returns:
            torch.Tensor: Output tensor after processing through the Transformer model.
        """

        for block in self.blocks:
            x = block(x, attn_kwargs=attn_kwargs)

        return x
