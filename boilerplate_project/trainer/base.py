#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn.functional as F

from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import TrainerResult


class BoilerplateTrainer(BaseTrainer):
    """A base trainer implementation for the `ksuit` framework that runs a simple forward pass and computes a loss value."""

    @staticmethod
    def train_step(batch: dict[str, torch.Tensor], dist_model: torch.nn.Module) -> TrainerResult:
        """Forward method of the trainer that runs a forward pass on the model and computes the loss.

        Args:
            batch: dict with tensors for the forward pass and the loss computation.
            model: Model instance to run the forward pass on.

        Returns:
            TrainerResult containing the total loss.
        """
        # prepare data
        x = batch["x"]
        target = batch["y"]

        # forward
        y_hat = dist_model(x)

        # calculate loss
        loss = F.cross_entropy(y_hat, target)

        return TrainerResult(total_loss=loss)
