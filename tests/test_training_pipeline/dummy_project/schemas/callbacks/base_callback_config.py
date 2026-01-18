#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.schemas.callbacks import CallBackBaseConfig


class BaseCallbackConfig(CallBackBaseConfig):
    kind: str | None = None
    dataset_key: str
    name: Literal["BaseCallback"] = "BaseCallback"
