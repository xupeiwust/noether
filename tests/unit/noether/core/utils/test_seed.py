#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import random
from unittest.mock import patch

import numpy as np
import torch

from noether.core.utils.seed import set_seed


def test_set_seed_calls_libraries_cpu():
    """Test that all libraries are seeded, and CUDA is skipped if unavailable."""
    seed_val = 123

    # Patch all the underlying seed functions:
    with (
        patch("random.seed") as mock_random,
        patch("numpy.random.seed") as mock_np,
        patch("torch.manual_seed") as mock_torch,
        patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
        patch("torch.cuda.is_available", return_value=False),
    ):
        set_seed(seed_val)

        # Verify calls:
        mock_random.assert_called_once_with(seed_val)
        mock_np.assert_called_once_with(seed_val)
        mock_torch.assert_called_once_with(seed_val)

        # Verify CUDA was skipped:
        mock_cuda_seed.assert_not_called()


def test_set_seed_calls_libraries_cuda():
    """Test that CUDA seeding is called when available."""
    seed_val = 123

    with (
        patch("random.seed") as mock_random,
        patch("numpy.random.seed") as mock_np,
        patch("torch.manual_seed") as mock_torch,
        patch("torch.cuda.manual_seed_all") as mock_cuda_seed,
        patch("torch.cuda.is_available", return_value=True),
    ):
        set_seed(seed_val)

        # Verify CUDA was called:
        mock_cuda_seed.assert_called_once_with(seed_val)


def test_set_seed_reproducibility():
    """
    Verify that calling set_seed actually produces deterministic results
    across Python, Numpy, and PyTorch.
    """
    seed_val = 999

    # --- Python Random
    set_seed(seed_val)
    val1_py = random.random()
    val1_list = [random.randint(0, 100) for _ in range(3)]

    set_seed(seed_val)
    val2_py = random.random()
    val2_list = [random.randint(0, 100) for _ in range(3)]

    assert val1_py == val2_py
    assert val1_list == val2_list

    # --- Numpy
    set_seed(seed_val)
    val1_np = np.random.rand(5)

    set_seed(seed_val)
    val2_np = np.random.rand(5)

    assert np.allclose(val1_np, val2_np)

    # --- PyTorch
    set_seed(seed_val)
    val1_torch = torch.randn(5)

    set_seed(seed_val)
    val2_torch = torch.randn(5)

    assert torch.equal(val1_torch, val2_torch)
