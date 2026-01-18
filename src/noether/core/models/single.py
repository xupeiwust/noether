#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import torch

from noether.core.factory import OptimizerFactory
from noether.core.models.base import ModelBase

if TYPE_CHECKING:  # import only for type checking to avoid circular imports
    from torch.amp.grad_scaler import GradScaler

    from noether.core.providers import PathProvider
    from noether.core.schemas.models import ModelBaseConfig
    from noether.core.utils.training import UpdateCounter
    from noether.data.container import DataContainer


class Model(ModelBase):
    def __init__(
        self,
        model_config: ModelBaseConfig,
        is_frozen: bool = False,
        update_counter: UpdateCounter | None = None,
        path_provider: PathProvider | None = None,
        data_container: DataContainer | None = None,
        static_context: dict[str, Any] | None = None,
    ):
        """Base class for single models, i.e. one model with one optimizer as opposed to CompositeModel.

        Args:
            model_config: Model configuration.
            is_frozen: If true, will set `requires_grad` of all parameters to false. Will also put the model into eval
                mode (e.g., to put Dropout or BatchNorm into eval mode).
            path_provider: A path provider used by the initializer to store or retrieve checkpoints.
            data_container: The data container which includes the data and dataloader.
                This is currently unused but helpful for quick prototyping only, evaluating forward in debug mode, etc.
            static_context: The static context used to pass information between submodules, e.g. patch_size, latent_dim.
        """
        super().__init__(
            model_config=model_config,
            update_counter=update_counter,
            path_provider=path_provider,
            data_container=data_container,
            initializer_config=model_config.initializers,  # type: ignore[arg-type]
            static_context=static_context,
        )

        self._device = torch.device("cpu")

        self._optimizer_constructor = OptimizerFactory().create(
            model_config.optimizer_config
        )  # the OptimFactory creates a partial function
        self._is_frozen = is_frozen

        # check parameter combinations
        if self.is_frozen and self._optimizer_constructor is not None:
            raise ValueError("model.is_frozen=True but model.optimizer_constructor is not None")

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def device(self) -> torch.device:
        return self._device

    def get_named_models(self) -> dict[str, ModelBase]:
        """Returns a dict of {model_name: model}, e.g., to log all learning rates of all models/submodels."""
        return {self.name: self}

    def initialize_weights(self) -> Self:
        """Freezes the weights of the model by setting requires_grad to False if self.is_frozen is True."""
        if self.is_frozen:
            # frozen modules are by default used in eval mode; relevant for batchnorm, dropout, stochastic depth, etc.
            self.logger.info(f"{self.name} is frozen -> put in eval mode")
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        return self

    def apply_initializers(self) -> Self:
        """Apply the initializers to the model, calling initializer.init_weights and initializer.init_optim."""
        for initializer in self.initializers:
            initializer.init_weights(self)
            initializer.init_optimizer(self)
        return self

    def initialize_optimizer(self) -> None:
        """Initialize the optimizer."""
        if self._optimizer_constructor is not None:
            self._optim = self._optimizer_constructor(self, update_counter=self.update_counter)
            self.logger.info(f"Initialized {self._optim} optimizer for {self.name}")
        elif not self.is_frozen:
            if self.trainable_param_count == 0:
                self.is_frozen = True  # type: ignore[misc]
                self.logger.info(f"{self.name} has no trainable parameters -> freeze and put into eval mode")
                self.eval()
            else:
                raise RuntimeError(f"no optimizer for {self.name} and it's also not frozen")
        else:
            self.logger.info(f"{self.name} is frozen -> no optimizer to initialize")

    def optimizer_step(self, grad_scaler: GradScaler | None) -> None:
        """Perform an optimization step."""
        if self._optim is not None:
            self._optim.step(grad_scaler)

    def optimizer_schedule_step(self) -> None:
        """Perform the optimizer learning rate scheduler step."""
        if self._optim is not None:
            self._optim.schedule_step()

    def optimizer_zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of the optimizer."""
        if self._optim is not None:
            self._optim.zero_grad(set_to_none)

    def train(self, mode: bool = True) -> Self:
        """Set the model to train or eval mode.

        Overwrites the nn.Module.train method to avoid setting the model to train mode if it is frozen.

        Args:
            mode: If True, set the model to train mode. If False, set the model to eval mode.
        """
        # avoid setting mode to train if whole network is frozen
        # this prevents the training behavior of e.g. the following components
        # - Dropout/StochasticDepth dropping during
        # - BatchNorm (in train mode the statistics are tracked)
        if self.is_frozen and mode is True:
            if self.training:
                self.logger.error(
                    f"model {type(self).__name__} is in train mode but it is frozen and "
                    "shouldn't be used in train mode -> put it back to eval mode"
                )
            return super().eval()
        return super().train(mode=mode)

    def to(self, device: str | torch.device | int | None, *args, **kwargs) -> Self:  # type: ignore[override]
        """Performs Tensor dtype and/or device conversion, overwriting nn.Module.to method to set the _device attribute.

        Args:
            device: The desired device of the tensor. Can be a string (e.g. "cuda:0") or "cpu".
        """
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e:
                self.logger.error(f"Invalid device: {device}")
                raise e
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device)}")
        self._device = device
        return super().to(*args, **kwargs, device=device)  # type: ignore[no-any-return,call-overload]
