#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast


class WandBMode(StrEnum):
    """
    Enumeration of WandB run modes.

    ONLINE: Normal WandB mode, logs are synced to the server in real-time.
    OFFLINE: Logs are stored locally and can be synced later.
    DISABLED : WandB is disabled entirely; no logs are created.
    """

    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"

    @property
    def value(self) -> Literal["online", "offline", "disabled"]:
        """Return the mode value as a Literal type for static type checkers."""
        return cast("Literal['online', 'offline', 'disabled']", super().value)


@dataclass(frozen=True, slots=True)
class WandBConfig:
    """
    Immutable configuration object for a WandB run.

    Attributes:
        mode: The WandB execution mode (ONLINE, OFFLINE, or DISABLED).
        host: The WandB server hostname (required unless DISABLED).
        entity: The WandB entity (user or team name) (required unless DISABLED).
        project: The WandB project name (required unless DISABLED).

    This class enforces that `host`, `entity`, and `project` must be provided and must be valid strings unless
    the mode is DISABLED.
    """

    mode: WandBMode
    host: str | None = field(default=None)
    entity: str | None = field(default=None)
    project: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not self.is_online:
            return

        for name, value in (("host", self.host), ("entity", self.entity), ("project", self.project)):
            if not isinstance(value, str):
                raise TypeError(f"WandB {name} must be a valid string, got {value!r}")

    @property
    def is_disabled(self) -> bool:
        """Return True if WandB mode is DISABLED."""
        return self.mode is WandBMode.DISABLED

    @property
    def is_offline(self) -> bool:
        """Return True if WandB mode is OFFLINE."""
        return self.mode is WandBMode.OFFLINE

    @property
    def is_online(self) -> bool:
        """Return True if WandB mode is ONLINE."""
        return self.mode is WandBMode.ONLINE

    def require_online(self) -> None:
        """Raise a RuntimeError if WandB is not in ONLINE mode.

        Use this in code paths that require a live WandB connection.
        """
        if not self.is_online:
            raise RuntimeError("WandB must be in ONLINE mode for this operation.")
