#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.models import UPTConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.modules import DeepPerceiverDecoder, SupernodePooling, TransformerBlock
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.mlp import MLP


class UPT(nn.Module):
    """Implementation of the UPT (Universal Physics Transformer) model."""

    def __init__(
        self,
        config: UPTConfig,
    ):
        """

        Args:
            config: Configuration for the UPT model.
        """

        super().__init__()

        self.encoder = SupernodePooling(config=config.supernode_pooling_config)
        self.use_rope = config.use_rope

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(
                hidden_dim=config.decoder_config.perceiver_block_config.hidden_dim,
                input_dim=config.data_specs.position_dim,
            )  # type: ignore[call-arg]
        )

        if self.use_rope:
            if not config.approximator_config.use_rope and config.decoder_config.perceiver_block_config.use_rope:
                raise ValueError(
                    "If 'use_rope' is set to True in the UPTConfig, it must also be set to True in the encoder_config."
                )
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=config.hidden_dim // config.num_heads,
                    input_dim=config.data_specs.position_dim,
                    implementation="complex",
                )  # type: ignore[call-arg]
            )

        self.approximator_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=config.approximator_config,
                )
                for _ in range(config.approximator_depth)
            ],
        )
        self.decoder = DeepPerceiverDecoder(config=config.decoder_config)

        self.norm = nn.LayerNorm(
            config.decoder_config.perceiver_block_config.hidden_dim,
            eps=config.decoder_config.perceiver_block_config.eps,
        )
        self.prediction_layer = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.decoder_config.perceiver_block_config.hidden_dim,
                output_dim=config.data_specs.total_output_dim,
                init_weights=config.decoder_config.perceiver_block_config.init_weights,
            )  # type: ignore[call-arg]
        )
        self.bias_layers = False
        if config.bias_layers:
            self.bias_layers = True
            self.surface_bias = MLP(
                config=MLPConfig(
                    input_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                )
            )

            self.volume_bias = MLP(
                config=MLPConfig(
                    input_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                )
            )

    def surface_and_volume_bias(self, x: torch.Tensor, surface_mask: torch.Tensor) -> torch.Tensor:
        """Apply separate bias layers for surface and volume points."""
        # TODO: This method needs to be removed after we refactor the tutorial. We need one module that does
        #  the biasing for all of our models, not an implementation in each model.
        unbatch = False
        if x.ndim == 2:
            # if we have a single point, we need to add a batch dimension
            unbatch = True
            x = x.unsqueeze(0)

        surface_mask = surface_mask[0]  #
        x_surface = self.surface_bias(x[:, surface_mask.bool(), :])
        x_volume = self.volume_bias(x[:, ~surface_mask.bool(), :])
        x = torch.concat([x_surface, x_volume], dim=1)
        if unbatch:
            x = x.squeeze(0)
        return x

    def compute_rope_args(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        query_position: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute the RoPE frequency arguments for the surface_position and query_position. If we don't use RoPE,
        return empty dicts.

        Args:
            surface_position_batch_idx: Batch indices for the surface positions.
            surface_position: Surface position coordinates.
            surface_position_supernode_idx: Supernode indices for the surface positions.
            query_position: Query position coordinates.
        Returns:
            dict[str, torch.Tensor]: Dictionary containing the RoPE frequency arguments.
        """
        if not self.use_rope:
            return {}, {}

        # supernode pooling needs a sparse tensor. However, for the output after the supernode pooling,
        # we need to know the cast the shapes back to the batch dimension.
        batch_size = surface_position_batch_idx.unique().shape[0]
        supernode_freqs = self.rope(surface_position[surface_position_supernode_idx])
        channels = supernode_freqs.shape[-1]
        if supernode_freqs.ndim == 2:
            supernode_freqs = supernode_freqs.unsqueeze(0)  # add batch dimension
        # bring back the batch dimension
        supernode_freqs = supernode_freqs.reshape(batch_size, -1, channels)
        encoder_attn_kwargs = dict(freqs=supernode_freqs)
        decoder_attn_kwargs = dict(
            q_freqs=self.rope(query_position),
            k_freqs=supernode_freqs,
        )

        return encoder_attn_kwargs, decoder_attn_kwargs

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        query_position: torch.Tensor,
        surface_mask_query: torch.Tensor,
        input_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the UPT model.

        Args:
            input_position: input coordinates of the surface points.
            surface_mask_query: surface mask for the query points, indicating which points are surface points.
            surface_position_batch_idx: batch indices for the surface positions, since the surface positions are
                a sparse tensor for the supernode pooling.
            surface_position_supernode_idx: supernode indices for the surface positions.
            geometry_position: geometry position information.
            query_position: input coordinates of the query points.
            surface_features: surface features for the input points. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with the output tensors, containing the surface pressure and volume
            velocity.
        """

        encoder_attn_kwargs, decoder_attn_kwargs = self.compute_rope_args(
            surface_position_batch_idx, surface_position, surface_position_supernode_idx, query_position
        )

        # supernode pooling encoder
        x = self.encoder(
            input_pos=surface_position,
            supernode_idx=surface_position_supernode_idx,
            batch_idx=surface_position_batch_idx,
            input_features=input_features,
        )
        # approximator blocks
        for block in self.approximator_blocks:
            x = block(x, attn_kwargs=encoder_attn_kwargs)

        queries = self.pos_embed(query_position)

        if self.bias_layers:
            queries = self.surface_and_volume_bias(queries, surface_mask_query)

        # perceiver decoder
        x = self.decoder(
            kv=x,
            queries=queries,
            attn_kwargs=decoder_attn_kwargs,
            condition=None,
        )

        x = self.norm(x)
        x = self.prediction_layer(x)

        return x
