"""APScheduler wiring for recurring pipeline runs.

The scheduler configuration (daily / weekly / interval) lives in
:class:`SchedulerSettings`; this module only maps it to an APScheduler trigger
and runs the pipeline on that schedule until interrupted. APScheduler is
imported lazily so single runs and the unit tests do not require it.
"""

from __future__ import annotations

from typing import Any

from src.config.logging_config import get_logger
from src.config.settings import SchedulerSettings, Settings
from src.pipeline.pipeline import run_pipeline
from src.utils.errors import PlatformError


def build_trigger(settings: SchedulerSettings) -> Any:
    """Create the APScheduler trigger for the configured schedule.

    Raises
    ------
    ConfigurationError
        Propagated by APScheduler when the timezone is unknown.
    """
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    if settings.mode == "interval":
        return IntervalTrigger(
            minutes=settings.interval_minutes, timezone=settings.timezone
        )
    hour, minute = settings.time_of_day.hour, settings.time_of_day.minute
    if settings.mode == "weekly":
        return CronTrigger(
            day_of_week=settings.day_of_week,
            hour=hour,
            minute=minute,
            timezone=settings.timezone,
        )
    return CronTrigger(hour=hour, minute=minute, timezone=settings.timezone)


def run_scheduler(
    settings: Settings,
    *,
    dry_run: bool = False,
    use_demo_data: bool | None = None,
    sheets_service: Any | None = None,
) -> None:
    """Block and run the pipeline on the configured schedule (Ctrl+C to stop)."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    logger = get_logger("scheduler")
    trigger = build_trigger(settings.scheduler)
    scheduler = BlockingScheduler(timezone=settings.scheduler.timezone)

    def _job() -> None:
        try:
            result = run_pipeline(
                settings,
                dry_run=dry_run,
                use_demo_data=use_demo_data,
                sheets_service=sheets_service,
            )
            logger.info("scheduled run %s finished with %s", result.run_id, result.outcome)
        except PlatformError as exc:
            logger.error("scheduled run failed: %s", exc)

    scheduler.add_job(
        _job,
        trigger,
        id="pipeline-run",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    logger.info("scheduler started: %s", settings.scheduler.describe())
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler stopped")


__all__ = ["build_trigger", "run_scheduler"]