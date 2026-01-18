#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.dataset import DatasetBaseConfig
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig  # TODO: fix import path


class AeroDatasetConfig(DatasetBaseConfig):
    pipeline: AeroCFDPipelineConfig
    filter_categories: tuple[str] | None = None
