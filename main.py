#!/usr/bin/env python3
"""Command-line entry point for the Google Sheets Business Automation Platform.

Usage examples::

    python main.py                  # one run (live when GS_SPREADSHEET_ID is set)
    python main.py --dry-run        # never write to Google Sheets
    python main.py --demo           # always use the synthetic demo dataset
    python main.py --schedule       # run continuously on the configured schedule

Exit codes: 0 = success, 1 = pipeline failure, 2 = configuration error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import Settings, load_settings
from src.pipeline import run_pipeline, run_scheduler
from src.utils.errors import PlatformError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsba",
        description=(
            "Run the business automation pipeline: read -> validate -> analyse "
            "-> AI interpretation -> workbook report."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the report locally instead of to Google Sheets",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use the built-in synthetic dataset instead of reading the spreadsheet",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="run continuously on the configured schedule instead of once",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging for this invocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings: Settings = load_settings()
    except PlatformError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.verbose:
        settings = replace(settings, logging=replace(settings.logging, level="DEBUG"))
    configure_logging(settings.logging)
    logger = get_logger("cli")

    use_demo = True if args.demo else None
    try:
        if args.schedule:
            run_scheduler(settings, dry_run=args.dry_run, use_demo_data=use_demo)
            return 0
        result = run_pipeline(settings, dry_run=args.dry_run, use_demo_data=use_demo)
    except PlatformError as exc:
        logger.error("%s", exc)
        return 1

    if result.succeeded:
        logger.info(
            "run %s succeeded in %.2fs: %s",
            result.run_id,
            result.duration_seconds,
            result.message,
        )
        return 0
    logger.error("run %s failed: %s", result.run_id, result.message)
    return 1


if __name__ == "__main__":
    sys.exit(main())