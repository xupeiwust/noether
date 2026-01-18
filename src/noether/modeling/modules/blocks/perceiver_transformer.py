#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any, Literal

import torch
from torch import nn

from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig

AttentionType = Literal[
    "dot_product",
    "perceiver",
    "transolver",
    "transolver_plusplus",
]


class PerceiverTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        transformer_attn_ctor: AttentionType = "dot_product",
        init_weights: str = "truncnormal002",
        mlp_hidden_dim: int | None = None,
        drop_path: float = 0.0,
    ):
        """Instantiates a block which contains a perciever and a transformer block.
        Args:
            hidden_dim: hidden Dimension of the transformer block.
            num_heads: Number of attention heads.
            mlp_hidden_dim: Hidden dim of the feed forward MLP after the self-attention. Defaults to None.
            init_weights: Initialization method for the weight matrixes of the network. Defaults to "truncnormal002".
        """
        super().__init__()
        # import here to avoid circular dependencies
        # (PerceiverTransformerBlockpair is alphabetically before TransformerBlock in __init__.py)
        from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock

        self.perceiver = PerceiverBlock(
            PerceiverBlockConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                drop_path=drop_path,
                init_weights=init_weights,  # type: ignore[arg-type]
            )  # type: ignore[call-arg,arg-type]
        )
        self.transformer = TransformerBlock(
            TransformerBlockConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                drop_path=drop_path,
                attention_constructor=transformer_attn_ctor,
                init_weights=init_weights,  # type: ignore[arg-type]
            )  # type: ignore[call-arg]
        )

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, transformer_attn_kwargs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Forward pass of the transformer block.
        Args:
            q: Input tensor with shape (batch_size, num_query_tokens, hidden_dim).
            kv: Input tensor with shape (batch_size, num_kv_tokens, hidden_dim).
            transformer_attn_kwargs: Dict with arguments for the attention of the transformer block (such as the
                attention mask). Defaults to None.
        Returns:
            Result with shape (batch_size, num_query_tokens, hidden_dim).
        """
        q = self.perceiver(q=q, kv=kv)
        q = self.transformer(q, attn_kwargs=transformer_attn_kwargs)
        return q
