#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.encoders import MlpEncoder
from noether.modeling.modules.layers import LinearProjection

from .expected_output import MLP_ENCODER


@pytest.fixture
def encoder():
    torch.manual_seed(42)
    input_dim = 16
    hidden_dim = 32
    return MlpEncoder(input_dim=input_dim, hidden_dim=hidden_dim)


def test_mlp_encoder_initialization(encoder):
    """Test if the MlpEncoder initializes correctly."""
    assert isinstance(encoder.layer, torch.nn.Sequential)
    assert len(encoder.layer) == 3
    assert isinstance(encoder.layer[0], LinearProjection)
    assert isinstance(encoder.layer[1], torch.nn.GELU)
    assert isinstance(encoder.layer[2], LinearProjection)


def test_mlp_encoder_wrong_input_values():
    """Test MlpEncoder with invalid input values."""
    with pytest.raises(AssertionError):
        MlpEncoder(input_dim=0, hidden_dim=32)
    with pytest.raises(AssertionError):
        MlpEncoder(input_dim=16, hidden_dim=0)


def test_mlp_encoder_forward_backward(encoder):
    torch.manual_seed(42)
    """Test the forward method of MlpEncoder."""
    input_tensor = torch.randn(8, 16)  # Batch size of 8, input_dim of 16
    output_tensor = encoder(input_tensor)

    assert output_tensor.shape == (8, 32)  # Batch size of 8, hidden_dim of 32
    assert not torch.isnan(output_tensor).any(), "Output contains NaN values"
    assert not torch.isinf(output_tensor).any(), "Output contains Inf values"
    assert torch.allclose(output_tensor, MLP_ENCODER, 1e-2), "Output value mismatch"

    output_tensor.sum().backward()  # Test if backpropagation works

    assert encoder.layer[0].project.weight.grad is not None, "Gradients should not be None"
    assert encoder.layer[2].project.weight.grad is not None, "Gradients should not be None"


def test_mlp_encoder_invalid_input():
    """Test MlpEncoder with invalid input dimensions."""
    input_dim = 16
    hidden_dim = 32
    encoder = MlpEncoder(input_dim=input_dim, hidden_dim=hidden_dim)

    invalid_input = torch.randn(8, 10)  # Mismatched input_dim
    with pytest.raises(RuntimeError):
        encoder(invalid_input)
