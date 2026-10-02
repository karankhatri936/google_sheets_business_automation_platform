"""Report models.

``WorkbookReport`` is a *pure data* description of what should end up in the
workbook. It contains no Google-specific code, which means the same report can be
written to Google Sheets, exported to CSV, or asserted in tests without any
mocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CellValue = Any

MONEY_PATTERN = "#,##0.00"
INTEGER_PATTERN = "#,##0"
PERCENT_PATTERN = '0.0"%"'

# Formatting is intentionally restrained: a bold header row, frozen headers and
# number formats. No decorative colours or themes.
HEADER_BACKGROUND = {"red": 0.93, "green": 0.94, "blue": 0.96}


@dataclass(frozen=True)
class CellFormat:
    """Explicit number format for a single cell (0-based coordinates)."""

    row: int
    column: int
    pattern: str = MONEY_PATTERN
    number_type: str = "NUMBER"


@dataclass(frozen=True)
class SheetTable:
    """One worksheet worth of values plus its formatting hints."""

    name: str
    rows: tuple[tuple[CellValue, ...], ...]
    header_rows: tuple[int, ...] = (0,)
    freeze_rows: int = 1
    number_formats: tuple[CellFormat, ...] = ()
    column_widths: tuple[tuple[int, int], ...] = ()
    wrap_columns: tuple[int, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def values(self) -> list[list[CellValue]]:
        """Mutable copy of the rows (the API client sends plain lists)."""
        return [list(row) for row in self.rows]

    def is_empty(self) -> bool:
        return not self.rows


@dataclass(frozen=True)
class WorkbookReport:
    """The complete set of tables for one pipeline run.

    ``tables`` are written by overwriting the worksheet; ``append_tables`` (only
    the Run_Log) are written by appending, so run history accumulates.
    """

    tables: tuple[SheetTable, ...]
    append_tables: tuple[SheetTable, ...] = field(default_factory=tuple)

    def names(self) -> tuple[str, ...]:
        return tuple(table.name for table in (*self.tables, *self.append_tables))

    def table(self, name: str) -> SheetTable:
        for table in (*self.tables, *self.append_tables):
            if table.name == name:
                return table
        raise KeyError(f"no table named {name!r} in the report")

    def has_table(self, name: str) -> bool:
        return any(table.name == name for table in (*self.tables, *self.append_tables))

    @property
    def report_tables(self) -> tuple[SheetTable, ...]:
        """The report tables (all except the Run_Log)."""
        return self.tables

    @property
    def run_log_table(self) -> SheetTable | None:
        """The Run_Log table if present."""
        return self.append_tables[0] if self.append_tables else None

    @property
    def all_tables(self) -> tuple[SheetTable, ...]:
        """All tables including the Run_Log."""
        return self.tables + self.append_tables


# Backwards-compatible alias: sinks and the reporting package export use the
# ``ReportWorkbook`` name, while the report builder produces ``WorkbookReport``.
ReportWorkbook = WorkbookReport
