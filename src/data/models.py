"""Typed results produced by the data quality stages.

These dataclasses are intentionally plain (no pandas internals leaking) so they
can be logged, written to spreadsheets, asserted in tests and serialised to
JSON without surprises.

Convention
----------
Raw data frames always carry an integer index whose *label equals the
spreadsheet row number* (row 1 = header, first data row = 2). This makes every
validation/cleaning decision traceable back to the cell range a business user
sees in Google Sheets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import pandas as pd

Severity = Literal["error", "warning"]

# ---------------------------------------------------------------------------
# Canonical issue codes (used in logs, the Data_Quality sheet and tests)
# ---------------------------------------------------------------------------
CODE_MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
CODE_MISSING_VALUE = "MISSING_VALUE"
CODE_INVALID_NUMBER = "INVALID_NUMBER"
CODE_INVALID_INTEGER = "INVALID_INTEGER"
CODE_INVALID_DATE = "INVALID_DATE"
CODE_INVALID_CATEGORY = "INVALID_CATEGORY"
CODE_OUT_OF_RANGE = "OUT_OF_RANGE"
CODE_DUPLICATE_ORDER = "DUPLICATE_ORDER_ID"
CODE_EMPTY_DATASET = "EMPTY_DATASET"
CODE_REVENUE_OUTLIER = "REVENUE_OUTLIER"
CODE_REVENUE_MISMATCH = "REVENUE_MISMATCH"


@dataclass(frozen=True)
class ValidationIssue:
    """One aggregated validation finding.

    Issues are aggregated per (code, column) with a count and a few examples so
    the report stays readable when thousands of rows share the same problem.
    """

    code: str
    severity: Severity
    column: str | None
    message: str
    count: int
    examples: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class RejectedRecord:
    """A record that cannot be used, kept for transparency (never discarded)."""

    source_row: int
    order_id: str | None
    codes: tuple[str, ...]
    reasons: tuple[str, ...]
    values: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def reason_text(self) -> str:
        return "; ".join(self.reasons)

    def as_row(self) -> tuple[Any, ...]:
        """Flat representation for the Data_Quality worksheet."""
        return (self.source_row, self.order_id or "", ", ".join(self.codes), self.reason_text())


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating the raw dataset against the business schema."""

    total_rows: int
    valid_rows: int
    rejected: tuple[RejectedRecord, ...]
    issues: tuple[ValidationIssue, ...]
    duplicate_order_ids: tuple[str, ...] = ()
    outlier_rows: tuple[int, ...] = ()
    missing_columns: tuple[str, ...] = ()

    @property
    def rejected_rows(self) -> int:
        return len(self.rejected)

    @property
    def error_count(self) -> int:
        return sum(issue.count for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(issue.count for issue in self.issues if issue.severity == "warning")

    @property
    def rejected_index(self) -> tuple[int, ...]:
        """Spreadsheet row numbers of every rejected record."""
        return tuple(record.source_row for record in self.rejected)

    @property
    def duplicate_rows(self) -> int:
        """Number of rows beyond the first occurrence of a duplicated order id."""
        return len(self.duplicate_order_ids)

    def issues_for_code(self, code: str) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code == code)

    def rows_with_code(self, code: str) -> int:
        return sum(issue.count for issue in self.issues_for_code(code))

    def summary_lines(self) -> tuple[str, ...]:
        """Short human readable summary used in logs and the Run_Log sheet."""
        lines = [
            f"rows read: {self.total_rows}",
            f"rows valid: {self.valid_rows}",
            f"rows rejected: {self.rejected_rows}",
            f"errors: {self.error_count} | warnings: {self.warning_count}",
        ]
        if self.missing_columns:
            lines.append(f"missing required columns: {', '.join(self.missing_columns)}")
        if self.duplicate_order_ids:
            lines.append(f"duplicate order ids: {len(self.duplicate_order_ids)}")
        if self.outlier_rows:
            lines.append(f"revenue outlier rows flagged for review: {len(self.outlier_rows)}")
        return tuple(lines)


@dataclass(frozen=True)
class CoercionFailures:
    """Cells whose text could not be converted to the declared column type.

    Maps are read-only by convention: ``numeric`` is ``{column: {row: raw_text}}``
    and ``dates`` is ``{row: raw_text}``. Keeping the original text means the
    Data_Quality sheet can show a business user exactly which cell is wrong.
    """

    numeric: Mapping[str, Mapping[int, str]]
    dates: Mapping[int, str]
    integers: Mapping[str, Mapping[int, str]]

    @property
    def is_empty(self) -> bool:
        return not any(self.numeric.values()) and not self.dates and not any(
            self.integers.values()
        )

    def failure_rows(self, column: str) -> frozenset[int]:
        rows = set(self.numeric.get(column, {}))
        rows.update(self.integers.get(column, {}))
        return frozenset(rows)


@dataclass(frozen=True)
class CleaningAction:
    """A single deterministic cleaning rule that changed the dataset."""

    name: str
    count: int
    detail: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned dataset plus a transparent list of what was changed."""

    data: pd.DataFrame
    actions: tuple[CleaningAction, ...]
    rows_in: int
    rows_out: int

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def action_count(self, name: str) -> int:
        return sum(action.count for action in self.actions if action.name == name)

    def summary_lines(self) -> tuple[str, ...]:
        lines = [f"rows in: {self.rows_in}", f"rows out: {self.rows_out}"]
        lines.extend(
            f"{action.name}: {action.count} ({action.detail})"
            for action in self.actions
            if action.count
        )
        return tuple(lines)
