#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.pipeline.sample_processors import SamplewiseNormalizationSampleProcessor


@pytest.fixture
def precollator() -> SamplewiseNormalizationSampleProcessor:
    return SamplewiseNormalizationSampleProcessor(item="pressure", low=[1.0], high=[2.0])


@pytest.fixture
def precollator_linear() -> SamplewiseNormalizationSampleProcessor:
    return SamplewiseNormalizationSampleProcessor(item="pressure", low=None, high=[2.0])


def test_call_normalizes_sample(precollator):
    sample = {
        "pressure": torch.tensor([[3.0], [5.0]]),
        "unchanged": torch.tensor([[7.0, 9.0]]),
    }
    new_sample = precollator(sample)

    assert torch.allclose(
        new_sample["pressure"],
        torch.tensor([[1.0], [2.0]]),
    )
    assert torch.all(new_sample["unchanged"] == sample["unchanged"])


def test_call_normalizes_sample_linear(precollator_linear):
    sample = {
        "pressure": torch.tensor([[3.0], [5.0]]),
        "unchanged": torch.tensor([[7.0, 9.0]]),
    }
    new_sample = precollator_linear(sample)

    assert torch.allclose(
        new_sample["pressure"],
        torch.tensor([[2.0 / 5.0 * 3.0], [2.0]]),
    )
    assert torch.all(new_sample["unchanged"] == sample["unchanged"])


def test_call_raises_key_error_for_missing_item(precollator):
    sample = {"unchanged": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
    with pytest.raises(KeyError):
        precollator(sample)
