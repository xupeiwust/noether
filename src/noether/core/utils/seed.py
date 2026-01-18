#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set the seed for random number generation for Python `random`, numpy,
    torch and torch.cuda, if available.

    Args:
        seed: Seed value.
    """
    logger.info(f"Seeding process RNG with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
