#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path

import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from noether.training.runners import HydraRunner


def test_train_pipeline_does_not_error():
    # The config_path should be the DIRECTORY containing your configs
    with initialize(version_base=None, config_path="dummy_project/configs", job_name="test"):
        # This will now correctly find 'base_experiment.yaml' inside 'dummy_project/configs'
        config = compose(config_name="base_experiment")

    config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    HydraRunner().run(config)


def test_train_pipeline_copies_code(tmp_path: Path):
    # The config_path should be the DIRECTORY containing your configs
    with initialize(version_base=None, config_path="dummy_project/configs", job_name="test"):
        # This will now correctly find 'base_experiment.yaml' inside 'dummy_project/configs'
        config = compose(config_name="base_experiment")
    config = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    config["store_code_in_output"] = True
    config["output_path"] = tmp_path.as_posix()
    HydraRunner().run(config)
    # Find the actual output directory (may have timestamp subdirectory)
    output_dirs = list(tmp_path.iterdir())
    assert len(output_dirs) == 1 and output_dirs[0].is_dir()
    assert (output_dirs[0] / "code.tar.gz").exists()
