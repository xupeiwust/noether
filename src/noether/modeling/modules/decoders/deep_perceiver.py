#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Contains a PerceiverDecoder implementation."""

from typing import Any

import torch
from torch import nn

from noether.core.schemas.modules.decoders import DeepPerceiverDecoderConfig
from noether.modeling.modules.blocks import PerceiverBlock


class DeepPerceiverDecoder(nn.Module):
    """A deep Perceiver decoder module. Can be configured with different number of layers and hidden dimensions.
    However, it should be noted that this layer is not a full-fledged Perceiver, since it only has a cross-attention mechanism.
    """

    def __init__(
        self,
        config: DeepPerceiverDecoderConfig,
    ):
        """
        Initialize the DeepPerceiverDecoder.

        Args:
            config: Configuration for the DeepPerceiverDecoder module.
        """
        super().__init__()

        # create query coordinate embeddings

        self.blocks = nn.ModuleList(
            [PerceiverBlock(config=config.perceiver_block_config) for _ in range(config.depth)],
        )

    def forward(
        self,
        kv: torch.Tensor,
        queries: torch.Tensor,
        unbatch_mask: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Latent tokens as dense tensor (batch_size, num_latent_tokens, dim).
            pos: Query positions (batch_size, num_output_pos, pos_dim).
            block_kwargs: Additional arguments for the block.
            unbatch_mask: Unbatch mask.

        Returns:
            The predictions as sparse tensor (batch_size * num_output_pos, num_out_values).
        """
        assert kv.ndim == 3  # batch_size, num_latent_tokens, dim
        assert queries.ndim == 3  # batch_size, num_output_queries, pos_dim

        # perceiver
        for block in self.blocks:
            queries = block(q=queries, kv=kv, attn_kwargs=attn_kwargs, condition=condition)

        return queries
