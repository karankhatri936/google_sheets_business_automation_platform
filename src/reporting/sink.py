"""Report sinks: local CSV workbook and live Google Sheets.

Both sinks accept the same data structure (:class:`ReportWorkbook`) so the
automation pipeline can write locally during tests, preview locally in dry-run
mode, and write to Google Sheets in production without touching different code paths.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.google_sheets.client import GoogleSheetsClient
from src.google_sheets.formatting import build_workbook_formatting_requests
from src.reporting.models import ReportWorkbook, SheetTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LocalWriteResult:
    """Outcome of dumping a workbook to disk."""

    destination: Path
    written_files: tuple[Path, ...]
    table_count: int


@dataclass(frozen=True, slots=True)
class SheetsWriteResult:
    """Outcome of updating Google Sheets."""

    spreadsheet_id: str
    sheets_created: tuple[str, ...]
    sheets_updated: tuple[str, ...]
    run_log_row_appended: bool
    format_requests_sent: int


# ---------------------------------------------------------------------------
# Local filesystem sink (CSV directory)
# ---------------------------------------------------------------------------
class LocalReportSink:
    """Writes each SheetTable to a named CSV file inside an output directory."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)

    def write(self, workbook: ReportWorkbook) -> LocalWriteResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for table in workbook.all_tables:
            filename = f"{table.name}.csv"
            destination = self.output_dir / filename
            _write_table_csv(table, destination)
            written.append(destination)
            logger.debug("Wrote %s (%d rows)", destination, table.row_count)
        return LocalWriteResult(
            destination=self.output_dir,
            written_files=tuple(written),
            table_count=len(written),
        )


def _write_table_csv(table: SheetTable, destination: Path) -> None:
    with open(destination, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in table.rows:
            writer.writerow(["" if cell is None else cell for cell in row])

# ---------------------------------------------------------------------------
# Google Sheets sink
# ---------------------------------------------------------------------------
class GoogleSheetsSink:
    """Overwrites report tables and appends to the run log."""

    def __init__(self, client: GoogleSheetsClient, *, apply_formatting: bool = True) -> None:
        self.client = client
        self.apply_formatting = apply_formatting

    def write(
        self,
        spreadsheet_id: str,
        workbook: ReportWorkbook,
    ) -> SheetsWriteResult:
        existing_sheets = set(self.client.list_worksheets())
        sheet_ids = dict(self.client.worksheet_ids())

        created: list[str] = []
        updated: list[str] = []

        # Ensure all required worksheets exist
        needed_titles = [table.name for table in workbook.all_tables]
        for title in needed_titles:
            if title not in existing_sheets:
                logger.info("Creating missing worksheet %r in %s", title, spreadsheet_id)
                sheet_ids[title] = self.client.ensure_worksheet(title)
                created.append(title)
                existing_sheets.add(title)

        # 1. Overwrite regular report tables (replace existing data)
        for table in workbook.report_tables:
            self._write_report_table(table)
            updated.append(table.name)

        # 2. Append to run log (never overwrite existing rows)
        appended_log = False
        if workbook.run_log_table is not None:
            self._append_run_log(workbook.run_log_table)
            appended_log = True
            updated.append(workbook.run_log_table.name)

        # 3. Apply formatting (batchUpdate)
        formats_applied = 0
        if self.apply_formatting:
            formats_applied = self._apply_formatting(workbook.all_tables, sheet_ids)

        return SheetsWriteResult(
            spreadsheet_id=spreadsheet_id,
            sheets_created=tuple(created),
            sheets_updated=tuple(updated),
            run_log_row_appended=appended_log,
            format_requests_sent=formats_applied,
        )

    def _write_report_table(self, table: SheetTable) -> None:
        self.client.clear_values(f"{table.name}!A:Z")
        values = [list(row) for row in table.rows]
        self.client.write_values(f"{table.name}!A1", values)
        logger.debug(
            "Overwrote %s!A1 with %d rows x %d cols",
            table.name,
            table.row_count,
            table.column_count,
        )

    def _append_run_log(self, table: SheetTable) -> None:
        existing_values = self.client.read_values(f"{table.name}!A1:A1")
        if not existing_values:
            header_row = [list(table.rows[0])]
            self.client.write_values(f"{table.name}!A1", header_row)
            logger.info("Initialized run log header in %s", table.name)

        data_rows = [list(row) for row in table.rows[1:]]
        if data_rows:
            self.client.append_values(f"{table.name}!A1", data_rows)
            logger.info("Appended %d run-log row(s) to %s", len(data_rows), table.name)

    def _apply_formatting(
        self,
        tables: Sequence[SheetTable],
        sheet_ids: dict[str, int],
    ) -> int:
        requests = build_workbook_formatting_requests(tuple(tables), sheet_ids)
        if not requests:
            return 0
        self.client.apply_batch_update(requests)
        logger.debug("Applied %d formatting requests", len(requests))
        return len(requests)


__all__ = [
    "GoogleSheetsSink",
    "LocalReportSink",
    "LocalWriteResult",
    "SheetsWriteResult",
]
