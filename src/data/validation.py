"""Deterministic validation against the business schema.

Validation answers three questions and never mutates the data:

1. which records are **invalid** (they must not reach analytics, but stay visible
   in the Data_Quality worksheet with the reason for every rejection),
2. which records **need cleanup** (repairable through a documented rule, e.g. a
   missing customer name or a revenue value that does not match quantity x price),
3. which records are **valid** as they are.

Only problems that are genuinely fatal raise :class:`ValidationError`
(a missing required column or a completely empty dataset) - everything else is
reported as structured findings.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.config.schema import BUSINESS_SCHEMA, REQUIRED_COLUMNS, SCHEMA_BY_NAME, ColumnSpec
from src.config.logging_config import get_logger
from src.data.cleaning import canonical_value
from src.data.models import (
    CODE_DUPLICATE_ORDER,
    CODE_INVALID_CATEGORY,
    CODE_INVALID_DATE,
    CODE_INVALID_INTEGER,
    CODE_INVALID_NUMBER,
    CODE_MISSING_VALUE,
    CODE_OUT_OF_RANGE,
    CODE_REVENUE_MISMATCH,
    CODE_REVENUE_OUTLIER,
    CoercionFailures,
    RejectedRecord,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from src.utils.errors import ValidationError

MAX_EXAMPLES = 3
DEFAULT_DUPLICATE_COLUMN = "order_id"


class ValidationPolicy:
    """Configurable thresholds for validation (never hard-coded in the engine)."""

    __slots__ = ("revenue_outlier_threshold", "duplicate_check_column")

    def __init__(
        self,
        revenue_outlier_threshold: float = 10_000.0,
        duplicate_check_column: str = DEFAULT_DUPLICATE_COLUMN,
    ) -> None:
        self.revenue_outlier_threshold = revenue_outlier_threshold
        self.duplicate_check_column = duplicate_check_column


class _IssueLog:
    """Aggregates findings per (code, severity, column) with bounded examples."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, Severity, str | None], dict[str, Any]] = {}

    def add(
        self,
        code: str,
        severity: Severity,
        column: str | None,
        message: str,
        example: str | None = None,
    ) -> None:
        key = (code, severity, column)
        entry = self._entries.setdefault(key, {"message": message, "count": 0, "examples": []})
        entry["count"] += 1
        examples: list[str] = entry["examples"]
        if example and len(examples) < MAX_EXAMPLES and example not in examples:
            examples.append(example)

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def issues(self) -> tuple[ValidationIssue, ...]:
        ordered = sorted(
            self._entries.items(),
            key=lambda item: (item[0][1] != "error", item[0][0], item[0][2] or ""),
        )
        return tuple(
            ValidationIssue(
                code=code,
                severity=severity,
                column=column,
                message=entry["message"],
                count=entry["count"],
                examples=tuple(entry["examples"]),
            )
            for (code, severity, column), entry in ordered
        )


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive for odd objects
        return value is None


def _format_range_issue(spec: ColumnSpec, value: float) -> str:
    if spec.minimum is not None and value < spec.minimum:
        return f"{spec.name}: {value:g} is below the documented minimum of {spec.minimum:g}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{spec.name}: {value:g} is above the documented maximum of {spec.maximum:g}"
    return f"{spec.name}: {value:g} is outside the allowed range"


def _check_coercion_failures(
    frame: pd.DataFrame,
    failures: CoercionFailures,
    warnings: _IssueLog,
    reject: Any,
) -> dict[str, set[int]]:
    """Report coercion failures and return ``{column: rows}`` already rejected."""
    failed_rows: dict[str, set[int]] = {}

    def mark(column: str, row: int) -> None:
        failed_rows.setdefault(column, set()).add(row)

    for row, raw in sorted(failures.dates.items()):
        if row in frame.index:
            reject(row, CODE_INVALID_DATE, f"date: {raw!r} is not an accepted date format")
            mark("date", row)
    for column, cells in sorted(failures.integers.items()):
        for row, raw in sorted(cells.items()):
            if row in frame.index:
                reject(row, CODE_INVALID_INTEGER, f"{column}: {raw!r} is not a whole number")
                mark(column, row)
    for column, cells in sorted(failures.numeric.items()):
        spec = SCHEMA_BY_NAME.get(column)
        for row, raw in sorted(cells.items()):
            if row not in frame.index:
                continue
            if spec is not None and spec.derived:
                # A derived column is recomputed by the cleaning stage, so a
                # broken source value is a warning rather than a rejection.
                warnings.add(
                    CODE_INVALID_NUMBER,
                    "warning",
                    column,
                    f"{column}: not numeric - the value is recalculated during cleaning",
                    example=f"row {row}: {raw!r}",
                )
            else:
                reject(row, CODE_INVALID_NUMBER, f"{column}: {raw!r} is not numeric")
                mark(column, row)
    return failed_rows


def _check_schema_rules(
    frame: pd.DataFrame,
    schema: Iterable[ColumnSpec],
    warnings: _IssueLog,
    reject: Any,
    already_failed: dict[str, set[int]],
) -> None:
    for spec in schema:
        if spec.name not in frame.columns:
            continue
        failed_here = already_failed.get(spec.name, set())
        for row, value in frame[spec.name].items():
            if row in failed_here:
                continue  # already rejected with a more specific reason
            if _is_missing(value):
                if spec.default_on_missing:
                    warnings.add(
                        CODE_MISSING_VALUE,
                        "warning",
                        spec.name,
                        f"{spec.name}: missing value replaced by the documented default "
                        f"{spec.default_on_missing!r}",
                        example=f"row {row}",
                    )
                elif spec.required:
                    reject(row, CODE_MISSING_VALUE, f"{spec.name}: value is missing")
                continue

            if spec.kind in ("numeric", "integer"):
                numeric_value = float(value)
                if (spec.minimum is not None and numeric_value < spec.minimum) or (
                    spec.maximum is not None and numeric_value > spec.maximum
                ):
                    reject(row, CODE_OUT_OF_RANGE, _format_range_issue(spec, numeric_value))
            elif spec.kind == "category" and canonical_value(spec.name, value) is None:
                allowed = ", ".join(sorted(spec.allowed_values or ()))
                reject(
                    row,
                    CODE_INVALID_CATEGORY,
                    f"{spec.name}: {value!r} is not a recognised value (allowed: {allowed})",
                )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _column_from_message(message: str) -> str | None:
    """Extract the column a message refers to (messages use a ``column:`` prefix)."""
    prefix = message.partition(":")[0].strip()
    return prefix if prefix in SCHEMA_BY_NAME or prefix == "date" else None


def _display(value: Any, limit: int = 80) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _order_id_of(frame: pd.DataFrame, row: int) -> str | None:
    if "order_id" not in frame.columns:
        return None
    value = frame.at[row, "order_id"]
    return None if _is_missing(value) else str(value)


def validate_frame(
    frame: pd.DataFrame,
    failures: CoercionFailures,
    policy: ValidationPolicy | None = None,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
) -> ValidationResult:
    """Validate a prepared + type-coerced frame.

    Raises
    ------
    ValidationError
        When the dataset is empty or a required column is missing entirely -
        conditions under which a run cannot produce meaningful results.
    """
    policy = policy or ValidationPolicy()
    schema = tuple(schema)
    logger = get_logger("validation")

    if frame.empty:
        raise ValidationError("the source range contains no data rows (only a header row was found)")
    missing_columns = tuple(name for name in REQUIRED_COLUMNS if name not in frame.columns)
    if missing_columns:
        raise ValidationError(
            "required column(s) missing from the source data: " + ", ".join(missing_columns)
        )

    errors = _IssueLog()
    warnings = _IssueLog()
    reasons: dict[int, list[tuple[str, str]]] = {}

    def reject(row: int, code: str, message: str) -> None:
        reasons.setdefault(int(row), []).append((code, message))
        errors.add(code, "error", _column_from_message(message), message, example=f"row {row}")

    already_failed = _check_coercion_failures(frame, failures, warnings, reject)
    _check_schema_rules(frame, schema, warnings, reject, already_failed)

    rejected_rows = set(reasons)
    valid_rows = [row for row in frame.index if row not in rejected_rows]

    duplicate_ids = _check_duplicates(frame, valid_rows, policy, warnings)
    outlier_rows = _check_revenue(frame, valid_rows, policy, warnings)

    rejected = tuple(
        RejectedRecord(
            source_row=int(row),
            order_id=_order_id_of(frame, row),
            codes=tuple(sorted({code for code, _ in reasons[row]})),
            reasons=tuple(message for _, message in reasons[row]),
            values=tuple(
                (str(column), _display(frame.at[row, column])) for column in frame.columns
            ),
        )
        for row in sorted(reasons)
    )

    error_issues = errors.issues()
    for issue in error_issues:
        logger.warning(
            "validation: %s on column %s (%d record(s)) - %s",
            issue.code,
            issue.column or "n/a",
            issue.count,
            issue.message,
        )
    for issue in warnings.issues():
        logger.info(
            "validation: %s on column %s (%d record(s)) - %s",
            issue.code,
            issue.column or "n/a",
            issue.count,
            issue.message,
        )

    total_rows = len(frame)
    result = ValidationResult(
        total_rows=total_rows,
        valid_rows=total_rows - len(rejected),
        rejected=rejected,
        issues=error_issues + warnings.issues(),
        duplicate_order_ids=duplicate_ids,
        outlier_rows=outlier_rows,
    )
    logger.info(
        "validation complete: %d valid / %d rejected of %d row(s)",
        result.valid_rows,
        result.rejected_rows,
        result.total_rows,
    )
    return result


def _check_duplicates(
    frame: pd.DataFrame,
    valid_rows: list[int],
    policy: ValidationPolicy,
    warnings: _IssueLog,
) -> tuple[str, ...]:
    """Detect duplicate business keys (warning level: the record is repairable)."""
    column = policy.duplicate_check_column
    if column not in frame.columns or not valid_rows:
        return ()
    order_ids = frame.loc[valid_rows, column]
    duplicate_mask = order_ids.duplicated(keep="first")
    if not bool(duplicate_mask.any()):
        return ()
    for row, value in zip(order_ids.index[duplicate_mask], order_ids[duplicate_mask]):
        warnings.add(
            CODE_DUPLICATE_ORDER,
            "warning",
            column,
            "duplicate order id - only the first occurrence is used for analytics",
            example=f"row {row}: {value!r}",
        )
    return tuple(sorted({str(value) for value in order_ids[duplicate_mask]}))


def _check_revenue(
    frame: pd.DataFrame,
    valid_rows: list[int],
    policy: ValidationPolicy,
    warnings: _IssueLog,
) -> tuple[int, ...]:
    """Flag revenue outliers and revenue values that do not match quantity x price."""
    if "revenue" not in frame.columns:
        return ()
    has_inputs = {"quantity", "unit_price"} <= set(frame.columns)
    outliers: list[int] = []

    for row in valid_rows:
        revenue = frame.at[row, "revenue"]
        if not _is_missing(revenue) and float(revenue) > policy.revenue_outlier_threshold:
            outliers.append(int(row))
            warnings.add(
                CODE_REVENUE_OUTLIER,
                "warning",
                "revenue",
                "revenue exceeds the configured outlier threshold "
                f"({policy.revenue_outlier_threshold:g}) - kept, but worth a manual review",
                example=f"row {row}: {float(revenue):.2f}",
            )
        if not has_inputs:
            continue
        quantity = frame.at[row, "quantity"]
        unit_price = frame.at[row, "unit_price"]
        if _is_missing(quantity) or _is_missing(unit_price):
            continue
        expected = round(float(quantity) * float(unit_price), 2)
        if _is_missing(revenue):
            warnings.add(
                CODE_MISSING_VALUE,
                "warning",
                "revenue",
                "revenue is blank - derived from quantity * unit_price",
                example=f"row {row}",
            )
        elif round(float(revenue), 2) != expected:
            warnings.add(
                CODE_REVENUE_MISMATCH,
                "warning",
                "revenue",
                "revenue does not match quantity * unit_price - recalculated during cleaning",
                example=f"row {row}: given {float(revenue):.2f} -> {expected:.2f}",
            )
    return tuple(outliers)
