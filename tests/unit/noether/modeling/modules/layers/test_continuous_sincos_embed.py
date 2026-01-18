#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbed

from .expected_output import CONTINUOUS_SINCOS_EMBED


def test_continuous_sincos_embed_init_valid():
    dim = 128
    ndim = 3
    max_wavelength = 10000

    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim, max_wavelength=max_wavelength)
    embed = ContinuousSincosEmbed(config)

    assert embed.hidden_dim == dim
    assert embed.omega.shape[0] == 21
    assert embed.input_dim == ndim
    assert embed.max_wavelength == max_wavelength
    assert embed.padding == (dim % ndim) + (((dim - (dim % ndim)) // ndim) % 2) * ndim
    assert embed.omega is not None


def test_continuous_sincos_embed_init_invalid_dim():
    dim = 3
    ndim = 4

    with pytest.raises(AssertionError):
        config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
        ContinuousSincosEmbed(config)


def test_continous_sincos_embed_forward():
    torch.manual_seed(42)
    dim = 16
    ndim = 3
    num_coords = 8
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
    embed = ContinuousSincosEmbed(config)

    coords = torch.rand(2, num_coords, ndim)
    out = embed(coords)

    assert out.shape == (2, num_coords, dim)
    assert torch.allclose(out, CONTINUOUS_SINCOS_EMBED, rtol=1e-4)


def test_continous_sincos_embed_forward_fp32forced():
    torch.manual_seed(42)
    dim = 16
    ndim = 3
    num_coords = 8
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
    embed = ContinuousSincosEmbed(config)

    coords = torch.rand(2, num_coords, ndim)
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        out_fp16 = embed(coords)
    out_fp32 = embed(coords)
    assert torch.equal(out_fp16, out_fp32)
