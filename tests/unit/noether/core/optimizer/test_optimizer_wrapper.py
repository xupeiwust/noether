#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from noether.core.optimizer.optimizer_wrapper import OptimizerWrapper
from noether.core.schemas.optimizers import OptimizerConfig
from noether.core.utils.training.counter import UpdateCounter
from noether.core.utils.training.training_iteration import TrainingIteration


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x


class TestOptimizerWrapper:
    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def update_counter(self):
        return UpdateCounter(
            updates_per_epoch=10,
            effective_batch_size=1,
            start_iteration=TrainingIteration(epoch=0, update=0, sample=0),
            end_iteration=TrainingIteration(epoch=10),
        )

    @pytest.fixture
    def basic_config(self):
        return OptimizerConfig(
            exclude_bias_from_weight_decay=True,
            exclude_normalization_params_from_weight_decay=True,
            param_group_modifiers_config=[],
            schedule_config=None,
            weight_decay_schedule=None,
            clip_grad_norm=None,
            clip_grad_value=None,
        )

    def test_initialization(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        # Check that optimizer is initialized
        assert optimizer.torch_optim is not None
        assert len(optimizer.torch_optim.param_groups) > 0

    def test_exclude_bias_and_norm_from_weight_decay(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        # Check that bias and normalization layers have weight_decay=0
        for group in optimizer.torch_optim.param_groups:
            if "bn" in group["name"] or "bias" in group["name"]:
                assert group["weight_decay"] == 0.0

    def test_param_group_merging(self, model, basic_config, update_counter):
        # Create test parameter groups
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n == "conv.weight"],
                "name": "conv.weight",
                "lr_scale": 1.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n == "fc.weight"],
                "name": "fc.weight",
                "lr_scale": 1.0,
            },
        ]

        # Test the merge function directly
        merged_groups, merged_names = optimizer_wrapper = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )._merge_groups_with_the_same_parameters(param_groups)

        # Parameters with the same properties should be merged
        assert len(merged_groups) <= len(param_groups)

    def test_has_param_with_grad(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        # Initially no gradients
        assert not optimizer._has_param_with_grad()

        # Add some gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        # Now there should be gradients
        assert optimizer._has_param_with_grad()

    def test_zero_grad(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        # Set gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        # Call zero_grad
        optimizer.zero_grad()

        # Check gradients are None
        for p in model.parameters():
            assert p.grad is None

    def test_state_dict(self, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        state_dict = optimizer.state_dict()

        # Check that param_idx_to_name is in the state dict
        assert "param_idx_to_name" in state_dict

    @patch("noether.core.optimizer.optimizer_wrapper.Bidict")
    def test_load_state_dict(self, mock_bidict, model, basic_config, update_counter):
        optimizer = OptimizerWrapper(
            model=model,
            torch_optim_ctor=lambda param_groups: torch.optim.SGD(param_groups, lr=0.01, weight_decay=0.001),
            optim_wrapper_config=basic_config,
            update_counter=update_counter,
        )

        # Create a mock state dict with param_idx_to_name
        mock_state_dict = {
            "param_idx_to_name": {0: "conv.weight", 1: "conv.bias"},
            "state": {0: {"momentum_buffer": torch.ones(3, 16, 3, 3)}},
            "param_groups": [{"params": [0, 1], "lr": 0.01}],
        }

        # Mock the bidict behavior
        mock_bidict_instance = MagicMock()
        mock_bidict.return_value = mock_bidict_instance

        # Test load_state_dict
        with patch.object(optimizer.torch_optim, "load_state_dict") as mock_load:
            optimizer.load_state_dict(mock_state_dict)
            mock_load.assert_called_once()
