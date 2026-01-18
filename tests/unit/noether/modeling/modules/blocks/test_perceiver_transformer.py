#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.modeling.modules.blocks import PerceiverTransformerBlock

from .expected_output import PERCEIVER_TRANSFORMER_BLOCK


def test_block_forward():
    torch.manual_seed(42)
    batch_size = 2
    num_tokens = 4
    dim = 8
    q = torch.randn(batch_size, num_tokens, dim)
    kv = torch.randn(batch_size, num_tokens, dim)
    block = PerceiverTransformerBlock(
        hidden_dim=dim,
        num_heads=2,
        mlp_hidden_dim=16,
    )
    output = block(q, kv)
    assert torch.allclose(output, PERCEIVER_TRANSFORMER_BLOCK, 1e-2)
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == q.shape, "Output shape should match the input query shape"
    kv = torch.randn(batch_size, num_tokens + 1, dim)
    output = block(q=q, kv=kv)
    assert output.shape == q.shape, "Output shape should match the input query shape"


def test_block_initialization():
    # Note: PerceiverTransformerBlock doesn't use config but creates blocks internally
    block = PerceiverTransformerBlock(
        hidden_dim=8,
        num_heads=2,
        mlp_hidden_dim=16,
    )
    assert isinstance(block, nn.Module), "Block should be an instance of nn.Module"
    assert hasattr(block, "transformer"), "Block should have a transformer attribute"
    assert hasattr(block, "perceiver"), "Block should have a perceiver attribute"
