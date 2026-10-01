"""End-to-end pipeline orchestration.

One run performs the documented sequence::

    read raw values  ->  quality stages  ->  KPI analytics
        ->  AI interpretation + classification  ->  build workbook  ->  write sink

Live mode reads from / writes to the configured Google spreadsheet; dry-run
mode writes the workbook to the local output directory (reading the demo
dataset when no spreadsheet is configured). The orchestrator owns run metadata
(id, timing, outcome) and converts every known failure into a failed
:class:`PipelineResult` instead of crashing the caller.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai.classifier import classify_feedback
from src.ai.interpreter import interpret_analytics
from src.ai.models import AIInsights, ClassificationSummary
from src.analytics.kpi_engine import compute_analytics
from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.data.demo_data import build_demo_dataset
from src.data.io import dataframe_from_values
from src.data.quality import run_quality_stages
from src.data.validation import ValidationPolicy
from src.google_sheets.client import GoogleSheetsClient
from src.reporting.models import WorkbookReport
from src.reporting.report_builder import build_report_workbook
from src.reporting.sink import (
    GoogleSheetsSink,
    LocalReportSink,
    LocalWriteResult,
    SheetsWriteResult,
)
from src.utils.errors import ConfigurationError, DataReadError, PlatformError

OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"

_RUN_ID_STAMP = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of one pipeline run (success or a reported failure)."""

    run_id: str
    outcome: str
    message: str
    mode: str  # "live" | "dry_run"
    data_source: str  # "google_sheets" | "demo"
    started_at: str
    finished_at: str
    duration_seconds: float
    rows_read: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    rows_cleaned: int = 0
    net_revenue: float | None = None
    insights: AIInsights | None = None
    feedback: ClassificationSummary | None = None
    workbook: WorkbookReport | None = None
    write_result: LocalWriteResult | SheetsWriteResult | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime(_RUN_ID_STAMP)
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def run_pipeline(
    settings: Settings,
    *,
    dry_run: bool = False,
    use_demo_data: bool | None = None,
    sheets_service: Any | None = None,
) -> PipelineResult:
    """Execute one full pipeline run.

    Parameters
    ----------
    dry_run:
        Force local output even when a spreadsheet is configured.
    use_demo_data:
        ``None`` (default) reads the spreadsheet when configured and falls back
        to the synthetic demo dataset otherwise; ``True``/``False`` force it.
    sheets_service:
        Injectable Google Sheets API service (tests pass ``FakeSheetsService``).
    """
    logger = get_logger("pipeline")
    run_id = _new_run_id()
    started_at = _utc_now()
    started_clock = time.perf_counter()

    live_sheets = bool(settings.google_sheets.spreadsheet_id) and not dry_run
    demo = (
        use_demo_data
        if use_demo_data is not None
        else not settings.google_sheets.is_configured
    )
    mode = "live" if live_sheets else "dry_run"
    data_source = "demo" if demo else "google_sheets"
    logger.info("run %s starting (mode=%s, source=%s)", run_id, mode, data_source)

    client: GoogleSheetsClient | None = None
    try:
        if not demo:
            client = _build_client(settings, sheets_service)

        raw_values, source_note = _read_raw_values(settings, client, demo)
        raw_frame = dataframe_from_values(raw_values)
        policy = ValidationPolicy(
            revenue_outlier_threshold=settings.pipeline.revenue_outlier_threshold
        )
        quality = run_quality_stages(raw_frame, policy=policy)
        analytics = compute_analytics(
            quality.clean_data,
            top_n=settings.pipeline.top_n_products,
            source=data_source,
        )
        insights = interpret_analytics(settings.ai, analytics)
        feedback = classify_feedback(settings.ai, quality.clean_data)

        workbook = build_report_workbook(
            settings=settings,
            cleaning=quality.cleaning,
            quality=quality,
            analytics=analytics,
            insights=insights,
            run_id=run_id,
            started_at=started_at,
            finished_at=_utc_now(),
            duration_seconds=round(time.perf_counter() - started_clock, 2),
            outcome=OUTCOME_SUCCESS,
            message=source_note,
        )

        if live_sheets:
            assert client is not None
            write_result: LocalWriteResult | SheetsWriteResult = GoogleSheetsSink(
                client
            ).write(settings.google_sheets.spreadsheet_id, workbook)
            _trim_run_log(
                client,
                settings.google_sheets.worksheets.run_log,
                settings.google_sheets.run_log_max_rows,
            )
            message = f"report written to spreadsheet ({source_note})"
        else:
            output_dir = Path(settings.pipeline.local_output_dir)
            write_result = LocalReportSink(output_dir).write(workbook)
            message = f"report written to {output_dir} ({source_note})"

        result = PipelineResult(
            run_id=run_id,
            outcome=OUTCOME_SUCCESS,
            message=message,
            mode=mode,
            data_source=data_source,
            started_at=started_at,
            finished_at=_utc_now(),
            duration_seconds=round(time.perf_counter() - started_clock, 2),
            rows_read=quality.validation.total_rows,
            rows_valid=quality.validation.valid_rows,
            rows_rejected=quality.validation.rejected_rows,
            rows_cleaned=quality.cleaning.rows_out,
            net_revenue=round(analytics.overall.net_revenue, 2),
            insights=insights,
            feedback=feedback,
            workbook=workbook,
            write_result=write_result,
        )
        if not live_sheets:
            _write_run_summary(Path(settings.pipeline.local_output_dir), result)
        logger.info(
            "run %s finished: %s in %.2fs (%d row(s) cleaned)",
            run_id,
            result.outcome,
            result.duration_seconds,
            result.rows_cleaned,
        )
        return result
    except PlatformError as exc:
        logger.error("run %s failed: %s", run_id, exc)
        return _failed_result(
            run_id, str(exc), mode, data_source, started_at, started_clock
        )
    except Exception as exc:  # noqa: BLE001 - the orchestrator reports everything
        logger.exception("run %s failed unexpectedly", run_id)
        return _failed_result(
            run_id,
            f"unexpected error: {exc}",
            mode,
            data_source,
            started_at,
            started_clock,
        )


def _failed_result(
    run_id: str,
    message: str,
    mode: str,
    data_source: str,
    started_at: str,
    started_clock: float,
) -> PipelineResult:
    return PipelineResult(
        run_id=run_id,
        outcome=OUTCOME_FAILED,
        message=message,
        mode=mode,
        data_source=data_source,
        started_at=started_at,
        finished_at=_utc_now(),
        duration_seconds=round(time.perf_counter() - started_clock, 2),
    )


def _build_client(settings: Settings, sheets_service: Any | None) -> GoogleSheetsClient:
    """Create the API client (lazily building the service unless injected)."""
    spreadsheet_id = settings.google_sheets.spreadsheet_id
    if not spreadsheet_id:
        raise ConfigurationError("GS_SPREADSHEET_ID is not configured")
    if sheets_service is None:
        from src.google_sheets.auth import build_sheets_service

        sheets_service = build_sheets_service(settings.google_sheets)
    return GoogleSheetsClient(sheets_service, spreadsheet_id)


def _read_raw_values(
    settings: Settings, client: GoogleSheetsClient | None, demo: bool
) -> tuple[list[list[Any]], str]:
    """Raw value grid plus a short source note (recorded in the Run_Log)."""
    if demo:
        return build_demo_dataset().to_values(), "synthetic demo dataset"
    if client is None:
        raise ConfigurationError("no Google Sheets client is available for reading")
    a1_range = (
        f"{settings.google_sheets.worksheets.raw_data}!"
        f"{settings.google_sheets.raw_data_range}"
    )
    values = client.read_values(a1_range)
    if not values:
        raise DataReadError(f"the range {a1_range} contains no data")
    return values, a1_range


def _trim_run_log(client: GoogleSheetsClient, sheet_name: str, max_rows: int) -> None:
    """Bound the Run_Log worksheet to ``max_rows`` (header + newest rows)."""
    if max_rows < 2:
        return
    values = client.read_values(sheet_name)
    if len(values) <= max_rows:
        return
    kept = [values[0], *values[-(max_rows - 1) :]]
    client.write_values(f"{sheet_name}!A1", kept)
    get_logger("pipeline").info(
        "trimmed %s from %d to %d row(s)", sheet_name, len(values), len(kept)
    )


def _write_run_summary(output_dir: Path, result: PipelineResult) -> None:
    """Write ``run_summary.json`` next to the local CSV workbook."""
    payload = {
        "run_id": result.run_id,
        "outcome": result.outcome,
        "message": result.message,
        "mode": result.mode,
        "data_source": result.data_source,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "rows_read": result.rows_read,
        "rows_valid": result.rows_valid,
        "rows_rejected": result.rows_rejected,
        "rows_cleaned": result.rows_cleaned,
        "net_revenue": result.net_revenue,
        "insights_status": result.insights.status.value if result.insights else None,
        "feedback_status": result.feedback.status.value if result.feedback else None,
        "tables": result.workbook.names() if result.workbook else (),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )