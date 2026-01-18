#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import time
from types import TracebackType
from typing import Self


class Stopwatch:
    """A stopwatch class to measure elapsed time."""

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._elapsed_seconds: list[float] = []
        self._lap_start_time: float | None = None

    def start(self) -> Stopwatch:
        """Start the stopwatch."""
        assert self._start_time is None, "can't start running stopwatch"
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the stopwatch and return the elapsed time since the last lap."""
        assert self._start_time is not None, "can't stop a stopped stopwatch"

        if self._lap_start_time is None:
            # First lap:
            lap_time = time.perf_counter() - self._start_time
        else:
            # This is the final lap
            lap_time = time.perf_counter() - self._lap_start_time

        self._elapsed_seconds.append(lap_time)
        self._start_time = None
        self._lap_start_time = None
        return self._elapsed_seconds[-1]

    def lap(self) -> float:
        """Record a lap time and return the elapsed time since the last lap."""
        assert self._start_time is not None, "lap requires stopwatch to be started"
        if self._lap_start_time is None:
            # First lap:
            lap_time = time.perf_counter() - self._start_time
        else:
            # This is the final lap:
            lap_time = time.perf_counter() - self._lap_start_time
        self._elapsed_seconds.append(lap_time)
        self._lap_start_time = time.perf_counter()
        return lap_time

    @property
    def last_lap_time(self) -> float:
        """Return the last lap time."""
        assert len(self._elapsed_seconds) > 0, "last_lap_time requires lap()/stop() to be called at least once"
        return self._elapsed_seconds[-1]

    @property
    def lap_count(self) -> int:
        """Return the number of laps recorded."""
        return len(self._elapsed_seconds)

    @property
    def average_lap_time(self) -> float:
        """Return the average lap time."""
        assert len(self._elapsed_seconds) > 0, "average_lap_time requires lap()/stop() to be called at least once"
        return sum(self._elapsed_seconds) / len(self._elapsed_seconds)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.stop()

    @property
    def elapsed_seconds(self) -> float:
        """Return the total elapsed time since the stopwatch was started."""
        assert self._start_time is None, "elapsed_seconds requires stopwatch to be stopped"
        assert len(self._elapsed_seconds) > 0, "elapsed_seconds requires stopwatch to have been started and stopped"
        return sum(self._elapsed_seconds)

    @property
    def elapsed_milliseconds(self) -> float:
        """Return the total elapsed time since the stopwatch was started in milliseconds."""
        return self.elapsed_seconds * 1000
