#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.modeling.modules.layers import LayerScale, UnquantizedDropPath
from noether.modeling.modules.mlp import UpActDownMlp


class IrregularNatBlock(nn.Module):
    """Neighbourhood Attention Transformer (NAT) block for irregular grids. Consists of a single NAT attention layer and a feedforward layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_ctor: type,
        mlp_hidden_dim: int | None = None,
        drop_path: float = 0.0,
        norm_ctor: type = nn.LayerNorm,
        layerscale: float | None = None,
        eps: float = 1e-6,
        init_weights: str = "truncnormal002",
    ):
        """Initializes a NAT block for irregular grids.

        Args:
            dim: Hidden dimension of the NAT attention block.
            num_heads: Number of attention heads.
            attn_ctor: Constructor of the attention module. Why this this not Nat Attention by default?
            mlp_hidden_dim: Hidden dimension of the FF MLP modules. Defaults to None.
            drop_path: Probability to drop a path (i.e, attention/FF module) during training. Defaults to 0.0.
            norm_ctor: Constructor of the activation normalization. Defaults to nn.LayerNorm.
            layerscale: Initial value of the layer scale module. Defaults to None.
            eps: Epsilon value for the LayerNorm module. Defaults to 1e-6.
            init_weights: Initialization method for the weight parameters. Defaults to "truncnormal002".
        """
        super().__init__()
        self.norm1 = norm_ctor(dim, eps=eps)
        self.attn = attn_ctor(
            dim=dim,
            num_heads=num_heads,
            init_weights=init_weights,
        )
        self.ls1 = LayerScale(hidden_dim=dim, init_scale=layerscale)  # type: ignore[call-arg]
        self.drop_path1 = UnquantizedDropPath(drop_prob=drop_path)  # type: ignore[call-arg]
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = UpActDownMlp(
            input_dim=dim,
            hidden_dim=mlp_hidden_dim or dim * 4,
            init_weights=init_weights,
        )  # type: ignore[call-arg]
        self.ls2 = LayerScale(hidden_dim=dim, init_scale=layerscale)  # type: ignore[call-arg]
        self.drop_path2 = UnquantizedDropPath(drop_prob=drop_path)  # type: ignore[call-arg]

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x: _description_
            pos: _description_

        Returns:
            _description_
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), pos=pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
