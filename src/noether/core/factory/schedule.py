#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from noether.core.factory.base import Factory
from noether.core.schemas.schedules import AnyScheduleConfig
from noether.core.utils.training import ScheduleWrapper


class ScheduleFactory(Factory):
    """Factory for creating schedules. Handles wrapping into ScheduleWrapper which handles update/epoch based
    scheduling. Additionally, populates the `effective_batch_size` and `updates_per_epoch` to avoid specifying it
    in the config.
    """

    def create(self, schedule_config: AnyScheduleConfig, **kwargs) -> ScheduleWrapper | None:  # type: ignore[override]
        """Creates a schedule if the schedule is specified as dictionary. If the schedule was already instantiated, it
        will simply return the existing schedule. If `obj_or_kwargs` is None, None is returned.

        Args:
            schedule_config: The schedule config or already instantiated schedule.
            **kwargs: Additional keyword arguments that are passed to the schedule constructor.
        Returns:
            The instantiated schedule.
        """
        if schedule_config is None:
            return None

        update_counter = kwargs.pop("update_counter", None)
        schedule = self.instantiate(schedule_config, **kwargs)

        return ScheduleWrapper(schedule=schedule, update_counter=update_counter, interval=schedule_config.interval)
