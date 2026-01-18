#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.data.pipeline.sample_processor import SampleProcessor


class ReplaceKeySampleProcessor(SampleProcessor):
    """Utility processor that replaces the key with multiple other keys."""

    def __init__(self, source_key: str, target_keys: set[str]):
        """

        Args:
            source_key: Key to be replaced.
            target_keys: List of keys where source_key should be replaced in.
        """
        self.source_key = source_key
        self.target_keys = target_keys

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Replaces a key in the batch with one or multiple other keys.
        Creates a new dictionary whose keys are duplicated but uses references to the values of the old dict.
        This avoids copying the data and at the same time does not modify this function's input.

        Args:
            input_sample: Dictionary of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with the source key replaced by the target keys.
        """

        output_sample = self.save_copy(input_sample)
        source_item = output_sample.pop(self.source_key)
        for target_key in self.target_keys:
            output_sample[target_key] = self.save_copy(source_item)
        return output_sample
