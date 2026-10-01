"""Structured logging configuration.

A single call to :func:`configure_logging` wires a console handler and (when
configured) a rotating file handler on the package root logger. Secrets are
never logged: the formatter only emits the logger name, level and message.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config.settings import LoggingSettings

LOGGER_NAME = "gsba"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 2_000_000
_BACKUP_COUNT = 3


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure and return the application logger.

    The function is idempotent: calling it twice does not duplicate handlers
    (handlers created by us are removed and replaced).
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.level.upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(component: str) -> logging.Logger:
    """Return a child logger for a component, e.g. ``get_logger("analytics")``."""
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
