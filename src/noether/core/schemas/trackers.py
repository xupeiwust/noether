#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field


class WandBTrackerSchema(BaseModel):
    kind: str = Field(default="noether.core.trackers.WandBTracker", frozen=True)
    entity: str | None = Field(None)
    """The entity name for the W&B project."""
    project: str | None = Field(None)
    """The project name for the W&B project."""
    mode: Literal["disabled", "online", "offline"] | None = Field(default="online")
    """he mode of W&B. Can be 'disabled', 'online', or 'offline'."""
