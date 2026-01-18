#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from noether.core.factory.dataset import DatasetFactory


def test_init_defaults():
    """Test default initialization of wrapper factory."""
    with patch("noether.core.factory.dataset.Factory") as MockBaseFactory:
        factory = DatasetFactory()
        # Should create a default Factory for wrappers if none provided:
        assert factory.dataset_wrapper_factory == MockBaseFactory.return_value


def test_instantiate_no_wrappers():
    """Test instantiation when config has no wrappers."""
    factory = DatasetFactory()

    mock_config = SimpleNamespace(dataset_wrappers=None, kind="MyDataset")

    # Mock super().instantiate behavior
    # We patch 'noether.core.factory.dataset.Factory.instantiate' which is the parent method
    with patch("noether.core.factory.dataset.Factory.instantiate") as mock_super_inst:
        mock_dataset = "base_dataset"
        mock_super_inst.return_value = mock_dataset

        result = factory.instantiate(mock_config, extra="arg")

        assert result == mock_dataset
        mock_super_inst.assert_called_once_with(mock_config)


def test_instantiate_with_wrappers():
    """Test that wrappers are applied recursively."""
    mock_wrapper_factory = MagicMock()
    factory = DatasetFactory(dataset_wrapper_factory=mock_wrapper_factory)

    wrapper1_cfg = SimpleNamespace(kind="Wrapper1")
    wrapper2_cfg = SimpleNamespace(kind="Wrapper2")
    mock_config = SimpleNamespace(dataset_wrappers=[wrapper1_cfg, wrapper2_cfg], kind="BaseDataset")

    with patch("noether.core.factory.dataset.Factory.instantiate") as mock_super_inst:
        # Base creation:
        base_dataset = "dataset_v0"
        mock_super_inst.return_value = base_dataset

        # First call returns v1, second call returns v2:
        mock_wrapper_factory.instantiate.side_effect = ["dataset_v1", "dataset_v2"]

        result = factory.instantiate(mock_config)

        assert result == "dataset_v2"
        assert mock_wrapper_factory.instantiate.call_count == 2

        mock_wrapper_factory.instantiate.assert_has_calls(
            [call(wrapper1_cfg, dataset="dataset_v0"), call(wrapper2_cfg, dataset="dataset_v1")]
        )


def test_instantiate_invalid_wrappers_type():
    """Test error when dataset_wrappers is not a list."""
    factory = DatasetFactory()
    mock_config = SimpleNamespace(dataset_wrappers="not_a_list", kind="ds")

    with patch("noether.core.factory.dataset.Factory.instantiate"):
        with pytest.raises(ValueError, match="must be a list"):
            factory.instantiate(mock_config)
