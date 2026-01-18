#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas import ConfigSchema

from .datasets.base_dataset_config import BaseDatasetConfig
from .models.base_model_config import BaseModelConfig
from .trainer.boilerplate_trainer_config import BoilerPlateTrainerConfig


class BoilerplateConfigSchema(ConfigSchema):
    model: BaseModelConfig
    trainer: BoilerPlateTrainerConfig
    datasets: dict[str, BaseDatasetConfig]
