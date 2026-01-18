#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.schemas.models import UPTConfig
from noether.modeling.models import UPT as UPTBackbone

from .base import BaseModel


class UPT(BaseModel):
    """Implementation of the UPT (Universal Physics Transformer) model.

    Args:
        BaseModel: Base model class that contains the utilities for all models we use in this tutorial.
    """

    def __init__(
        self,
        config: UPTConfig,
        **kwargs,
    ):
        """
        Args:
           config: Configuration of the UPT model.
        """

        super().__init__(model_config=config, **kwargs)

        self.upt_backbone = UPTBackbone(
            config=config,
        )

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_query_position: torch.Tensor,
        volume_query_position: torch.Tensor,
        surface_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the UPT model.

        Args:
            surface_mask_query: surface mask for the query points, indicating which points are surface points.
            surface_position_batch_idx: batch indices for the surface positions, since the surface positions are a sparse tensor for the supernode pooling.
            surface_position_supernode_idx: supernode indices for the surface positions.
            geometry_position: geometry position information.
            query_position: input coordinates of the query points.
            surface_query_position: surface query positions.
            volume_query_position: volume query positions.
            surface_features: surface features for the input points. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: dictionary with the output tensors, containing the surface pressure and volume velocity.
        """

        query_position = torch.cat([surface_query_position, volume_query_position], dim=1)
        surface_mask_query = torch.zeros(surface_query_position.shape[0], query_position.shape[1])
        surface_mask_query[:, : surface_query_position.shape[1]] = 1.0

        # add features to queries
        if surface_features is None:
            input_features = None
        else:
            input_features = surface_features.squeeze(0)  # remove batch dimension, since we only have one sample

        x = self.upt_backbone(
            surface_position_batch_idx=surface_position_batch_idx,
            surface_position_supernode_idx=surface_position_supernode_idx,
            surface_position=surface_position,
            query_position=query_position,
            input_features=input_features,
            surface_mask_query=surface_mask_query,
        )

        # assumes surface pressure on index 0, 1:3 volume velocity
        return self.gather_outputs(
            x=x,
            surface_mask=surface_mask_query,
        )
