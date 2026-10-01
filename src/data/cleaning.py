"""Deterministic data cleaning.

Every rule in this module is explicit, ordering-stable and testable. The engine
never calls an AI model: anything that can be done reliably with Python/pandas is
done here.

Pipeline stages (each a public function):

``prepare_frame``  -> drop blank rows, normalise headers and whitespace
``coerce_types``   -> convert text cells into dates / integers / numbers
``clean_frame``    -> apply the value rules (aliases, defaults, derived fields)

The stage order matters: validation runs *between* ``coerce_types`` and
``clean_frame`` so invalid records are reported before anything is rewritten.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime
from typing import Any

import pandas as pd

from src.config.schema import (
    ALL_COLUMNS,
    BUSINESS_SCHEMA,
    CATEGORY_ALIAS_MAP,
    DATE_FORMATS,
    SCHEMA_BY_NAME,
    ColumnSpec,
)
from src.data.io import is_blank
from src.data.models import (
    CleaningAction,
    CleaningResult,
    CoercionFailures,
    ValidationResult,
)

# Columns used for traceability in the cleaned output and the Clean_Data sheet.
SOURCE_ROW_COLUMN = "source_row"
# Derived column added by the cleaning stage (period analysis key).
ORDER_MONTH_COLUMN = "order_month"

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NUMERIC_NOISE_RE = re.compile(r"[^0-9.\-]")


# ---------------------------------------------------------------------------
# Action tracking (transparent reporting of every change)
# ---------------------------------------------------------------------------
class _ActionTracker:
    """Accumulates cleaning actions with counts and a few examples."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._details: dict[str, str] = {}
        self._examples: dict[str, list[str]] = {}

    def record(self, name: str, detail: str, *, count: int = 1, example: str | None = None) -> None:
        if count == 0:
            return
        self._counts[name] = self._counts.get(name, 0) + count
        self._details[name] = detail
        if example:
            examples = self._examples.setdefault(name, [])
            if len(examples) < 3 and example not in examples:
                examples.append(example)

    def actions(self) -> tuple[CleaningAction, ...]:
        return tuple(
            CleaningAction(
                name=name,
                count=count,
                detail=self._details[name],
                examples=tuple(self._examples.get(name, [])),
            )
            for name, count in self._counts.items()
        )


# ---------------------------------------------------------------------------
# Value level helpers (also used by the validation engine)
# ---------------------------------------------------------------------------
def normalise_text(value: Any) -> str | None:
    """Collapse whitespace and return ``None`` for blank values."""
    if is_blank(value):
        return None
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def header_key(header: Any) -> str:
    """Normalise a header spelling for alias matching (``Order ID`` -> ``order_id``)."""
    return _NON_ALNUM_RE.sub("_", str(header).strip().lower()).strip("_")


def canonical_value(column: str, raw: Any) -> str | None:
    """Map a categorical cell onto its canonical value.

    Returns ``None`` when the value cannot be mapped; the validation engine
    turns that into an ``INVALID_CATEGORY`` issue. Categorical values that are
    not in the whitelist are never guessed.
    """
    text = normalise_text(raw)
    if text is None:
        return None
    spec = SCHEMA_BY_NAME.get(column)
    aliases = CATEGORY_ALIAS_MAP.get(column, {})
    lookup_key = text.casefold()
    if lookup_key in aliases:
        return aliases[lookup_key]
    if spec is not None and spec.allowed_values is not None:
        for allowed in spec.allowed_values:
            if allowed.casefold() == lookup_key:
                return allowed
    return None


def parse_number(raw: Any) -> float | None:
    """Parse a business number such as ``"$1,299.00"`` or ``"(50)"``."""
    if is_blank(raw):
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return None if pd.isna(value) else value

    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    cleaned = _NUMERIC_NOISE_RE.sub("", text)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if pd.isna(value):
        return None
    return -value if negative else value


def parse_integer(raw: Any) -> int | None:
    """Parse a whole number; returns ``None`` for non-integers such as ``1.5``."""
    value = parse_number(raw)
    if value is None or not float(value).is_integer():
        return None
    return int(value)


def parse_date(raw: Any) -> datetime | None:
    """Parse a date using the explicit formats listed in :data:`DATE_FORMATS`.

    Ambiguous formats (``03/04/2025``) are intentionally rejected instead of
    being guessed - a wrong guess would silently corrupt period analysis.
    """
    if is_blank(raw):
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, date_type):
        return datetime(raw.year, raw.month, raw.day)
    text = str(raw).strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Stage 1 - preparation
# ---------------------------------------------------------------------------
def _header_lookup(schema: Iterable[ColumnSpec]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for spec in schema:
        for header in spec.accepted_headers:
            lookup.setdefault(header_key(header), spec.name)
    return lookup


def normalize_headers(
    frame: pd.DataFrame,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
    tracker: _ActionTracker | None = None,
) -> tuple[pd.DataFrame, tuple[CleaningAction, ...]]:
    """Map accepted header spellings onto canonical column names.

    Columns outside the schema are kept (renamed to snake_case) so unexpected
    data is never lost - they are simply not analysed.
    """
    tracker = tracker or _ActionTracker()
    lookup = _header_lookup(schema)
    mapping: dict[Any, str] = {}
    used: set[str] = set()

    for position, column in enumerate(frame.columns, start=1):
        canonical = lookup.get(header_key(column))
        if canonical is not None and canonical not in used:
            mapping[column] = canonical
            if canonical != column:
                tracker.record(
                    "headers_normalised",
                    "accepted header spellings mapped to canonical column names",
                    example=f"{column!r} -> {canonical!r}",
                )
        else:
            sanitized = header_key(column) or f"unnamed_column_{position}"
            while sanitized in used:
                sanitized = f"{sanitized}_2"
            mapping[column] = sanitized
            tracker.record(
                "unrecognised_columns_retained",
                "columns outside the business schema are kept but not analysed",
                example=str(column),
            )
        used.add(mapping[column])

    return frame.rename(columns=mapping), tracker.actions()


def prepare_frame(
    frame: pd.DataFrame,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
) -> tuple[pd.DataFrame, tuple[CleaningAction, ...]]:
    """Stage 1: blank rows, header names and whitespace.

    * rows where every cell is blank are dropped and counted
    * accepted header spellings are renamed to canonical column names
    * surrounding/duplicated whitespace is collapsed and blank tokens become NA
    """
    tracker = _ActionTracker()
    if frame.empty:
        return frame.copy(), ()

    work = frame.copy()

    blank_mask = work.apply(lambda row: all(is_blank(value) for value in row), axis=1)
    if bool(blank_mask.any()):
        rows = [str(row) for row in work.index[blank_mask][:3]]
        tracker.record(
            "blank_rows_dropped",
            "rows where every cell is blank carry no information",
            count=int(blank_mask.sum()),
            example=f"row {', '.join(rows)}",
        )
        work = work.loc[~blank_mask]

    work, header_actions = normalize_headers(work, schema, tracker)
    del header_actions  # actions are already registered on the shared tracker

    text_columns = {
        spec.name for spec in schema if spec.kind in ("text", "category") and spec.name in work.columns
    }
    changed_cells = 0
    for column in work.columns:
        series = work[column]
        if column in text_columns:
            normalised = series.map(normalise_text)
            for row, before, after in zip(series.index, series, normalised):
                if not is_blank(before) and after != before:
                    changed_cells += 1
                    if changed_cells <= 3:
                        tracker.record(
                            "whitespace_normalised",
                            "leading/trailing/duplicated whitespace collapsed in text values",
                            count=1,
                            example=f"row {row}: {before!r} -> {after!r}",
                        )
            work[column] = normalised
        else:
            work[column] = series.map(lambda value: None if is_blank(value) else value)

    if changed_cells > 3:
        tracker.record(
            "whitespace_normalised",
            "leading/trailing/duplicated whitespace collapsed in text values",
            count=changed_cells - 3,
        )

    return work, tracker.actions()


# ---------------------------------------------------------------------------
# Stage 2 - type coercion
# ---------------------------------------------------------------------------
def _failed_cells(series: pd.Series, parsed: pd.Series) -> dict[int, str]:
    """Rows where a non-blank cell could not be converted (row -> raw text).

    Note: ``Series.map`` may already have materialised the failures as ``NaN`` or
    ``NaT`` depending on the inferred dtype, so "unconvertible" is detected with
    ``pd.isna`` rather than an identity check against ``None``.
    """
    failures: dict[int, str] = {}
    for row, raw, value in zip(series.index, series, parsed):
        if bool(pd.isna(value)) and not is_blank(raw):
            failures[int(row)] = str(raw)
    return failures


def coerce_types(
    frame: pd.DataFrame,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
) -> tuple[pd.DataFrame, CoercionFailures]:
    """Stage 2: convert text cells into dates, integers and numbers.

    Conversion failures are *collected*, never dropped: each failed cell is
    returned with its original text so validation can reject the record and the
    Data_Quality sheet can show which cell a user has to fix.
    """
    work = frame.copy()
    numeric: dict[str, dict[int, str]] = {}
    integers: dict[str, dict[int, str]] = {}
    dates: dict[int, str] = {}

    for spec in schema:
        if spec.name not in work.columns:
            continue
        series = work[spec.name]

        if spec.kind == "date":
            parsed = series.map(parse_date)
            dates.update(_failed_cells(series, parsed))
            work[spec.name] = pd.to_datetime(parsed, errors="coerce")
        elif spec.kind == "integer":
            parsed = series.map(parse_integer)
            failures = _failed_cells(series, parsed)
            if failures:
                integers[spec.name] = failures
            work[spec.name] = pd.to_numeric(parsed, errors="coerce")
        elif spec.kind == "numeric":
            parsed = series.map(parse_number)
            failures = _failed_cells(series, parsed)
            if failures:
                numeric[spec.name] = failures
            work[spec.name] = pd.to_numeric(parsed, errors="coerce")
        else:
            work[spec.name] = series.map(normalise_text)

    failures = CoercionFailures(numeric=numeric, dates=dates, integers=integers)
    return work, failures



# ---------------------------------------------------------------------------
# Stage 3 - value rules
# ---------------------------------------------------------------------------
def _record_examples(
    tracker: _ActionTracker,
    name: str,
    detail: str,
    changes: list[tuple[int, str, str]],
    *,
    limit: int = 3,
) -> None:
    """Record a counted change with a bounded number of examples."""
    if not changes:
        return
    for row, before, after in changes[:limit]:
        tracker.record(name, detail, count=1, example=f"row {row}: {before!r} -> {after!r}")
    if len(changes) > limit:
        tracker.record(name, detail, count=len(changes) - limit)


def clean_frame(
    frame: pd.DataFrame,
    validation: ValidationResult,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
) -> CleaningResult:
    """Stage 3: apply the documented value rules to the *valid* records.

    Order of operations (each step is reported as a cleaning action):

    1. rejected records are removed (they stay visible in the Data_Quality sheet)
    2. duplicate order ids are removed, keeping the first occurrence
    3. categorical values are mapped onto canonical business values
    4. missing values are filled where a documented default exists
    5. ``revenue`` is recalculated as ``quantity * unit_price``
    6. derived period key (``order_month``) is added
    7. column order, dtypes and sorting are finalised
    """
    schema = tuple(schema)
    tracker = _ActionTracker()
    work = frame.copy()
    rows_in = len(work)

    rejected = [row for row in validation.rejected_index if row in work.index]
    if rejected:
        tracker.record(
            "rejected_rows_removed",
            "records that failed validation are excluded from analytics (details in Data_Quality)",
            count=len(rejected),
            example="rows " + ", ".join(str(row) for row in rejected[:3]),
        )
        work = work.drop(index=rejected)

    if "order_id" in work.columns and not work.empty:
        duplicate_mask = work["order_id"].duplicated(keep="first")
        duplicate_rows = [int(row) for row in work.index[duplicate_mask]]
        if duplicate_rows:
            tracker.record(
                "duplicate_orders_removed",
                "duplicate order ids removed; the first occurrence (by source row) is kept",
                count=len(duplicate_rows),
                example="rows " + ", ".join(str(row) for row in duplicate_rows[:3]),
            )
            work = work.loc[~duplicate_mask]

    for spec in schema:
        if spec.kind != "category" or spec.name not in work.columns:
            continue
        before = work[spec.name]
        after = before.map(lambda value, column=spec.name: canonical_value(column, value))
        changes = [
            (int(row), str(previous), str(current))
            for row, previous, current in zip(before.index, before, after)
            if not pd.isna(previous) and not pd.isna(current) and current != previous
        ]
        _record_examples(
            tracker,
            "categorical_values_normalised",
            "categorical values mapped onto canonical business values",
            changes,
        )
        work[spec.name] = after

    for spec in schema:
        if not spec.default_on_missing or spec.name not in work.columns:
            continue
        missing_mask = work[spec.name].isna()
        missing_count = int(missing_mask.sum())
        if missing_count:
            tracker.record(
                "default_values_filled",
                f"missing {spec.name!r} values replaced by the documented default",
                count=missing_count,
                example=f"{spec.name} = {spec.default_on_missing!r}",
            )
            work.loc[missing_mask, spec.name] = spec.default_on_missing

    if {"quantity", "unit_price", "revenue"} <= set(work.columns) and not work.empty:
        computed = (work["quantity"] * work["unit_price"]).round(2)
        provided = work["revenue"]
        mismatch_mask = provided.notna() & (provided.round(2) != computed)
        mismatch_rows = [
            (int(row), f"{previous:.2f}", f"{current:.2f}")
            for row, previous, current in zip(
                provided.index[mismatch_mask], provided[mismatch_mask], computed[mismatch_mask]
            )
        ]
        _record_examples(
            tracker,
            "revenue_recalculated",
            "revenue recomputed as quantity * unit_price (single source of truth for money metrics)",
            mismatch_rows,
        )
        missing_revenue = int(provided.isna().sum())
        if missing_revenue:
            tracker.record(
                "revenue_recalculated",
                "revenue recomputed as quantity * unit_price (single source of truth for money metrics)",
                count=missing_revenue,
                example="blank revenue cells were derived from quantity * unit_price",
            )
        work["revenue"] = computed

    if "date" in work.columns and not work.empty:
        work[ORDER_MONTH_COLUMN] = work["date"].dt.strftime("%Y-%m")
        tracker.record(
            "derived_columns_added",
            "order_month derived from date for period-over-period analysis",
            count=1,
        )

    if not work.empty:
        if "quantity" in work.columns:
            work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").astype("Int64")
        for column in ("unit_price", "revenue"):
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce").round(2)

        sort_columns = [column for column in ("date", "order_id") if column in work.columns]
        if sort_columns:
            work = work.sort_values(sort_columns, kind="stable")

    work.index = work.index.rename(SOURCE_ROW_COLUMN)
    work = work.reset_index()

    ordered = [
        column
        for column in (SOURCE_ROW_COLUMN, *ALL_COLUMNS, ORDER_MONTH_COLUMN)
        if column in work.columns
    ]
    extras = [column for column in work.columns if column not in ordered]
    work = work[ordered + extras]

    return CleaningResult(
        data=work,
        actions=tracker.actions(),
        rows_in=rows_in,
        rows_out=len(work),
    )
