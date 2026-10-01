"""Tests for the deterministic cleaning engine."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data.cleaning import (
    ORDER_MONTH_COLUMN,
    SOURCE_ROW_COLUMN,
    canonical_value,
    clean_frame,
    coerce_types,
    normalise_text,
    normalize_headers,
    parse_date,
    parse_integer,
    parse_number,
    prepare_frame,
)
from src.data.io import dataframe_from_values


# ---------------------------------------------------------------------------
# Value level helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,299.00", 1299.0),
        ("$1,299.00", 1299.0),
        ("(50)", -50.0),
        ("24.90", 24.9),
        (12, 12.0),
        ("abc", None),
        ("", None),
        ("N/A", None),
        (None, None),
        ("-", None),
    ],
)
def test_parse_number(raw: object, expected: float | None) -> None:
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [("3", 3), (5.0, 5), ("1.5", None), ("abc", None), ("", None)]
)
def test_parse_integer(raw: object, expected: int | None) -> None:
    assert parse_integer(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2025-02-03", datetime(2025, 2, 3)),
        ("03-Feb-2025", datetime(2025, 2, 3)),
        ("31/02/2025", None),
        ("03/02/2025", None),
        ("", None),
    ],
)
def test_parse_date_uses_explicit_formats(raw: str, expected: datetime | None) -> None:
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("column", "raw", "expected"),
    [
        ("category", " home and kitchen ", "Home & Kitchen"),
        ("category", "ELECTRONICS", "Electronics"),
        ("category", "Outdoors", "Outdoors"),
        ("region", "north", "North"),
        ("region", "Central", None),
        ("status", "completed", "Completed"),
        ("status", "Archived", None),
        ("status", "", None),
    ],
)
def test_canonical_value(column: str, raw: str, expected: str | None) -> None:
    assert canonical_value(column, raw) == expected


def test_normalise_text_collapses_whitespace() -> None:
    assert normalise_text("  Desk   Lamp  ") == "Desk Lamp"
    assert normalise_text("   ") is None


# ---------------------------------------------------------------------------
# Stage 1 - preparation
# ---------------------------------------------------------------------------
def test_normalize_headers_maps_accepted_spellings() -> None:
    frame = pd.DataFrame([{"Order Date": "2025-01-01", "Qty": 1, "Order Status": "Completed"}])
    renamed, actions = normalize_headers(frame)
    assert list(renamed.columns) == ["date", "quantity", "status"]
    assert any(action.name == "headers_normalised" for action in actions)


def test_normalize_headers_keeps_unknown_columns() -> None:
    frame = pd.DataFrame([{"Order Date": "2025-01-01", "Cost Centre": "EU-1"}])
    renamed, actions = normalize_headers(frame)
    assert "cost_centre" in renamed.columns
    assert any(action.name == "unrecognised_columns_retained" for action in actions)


def test_prepare_frame_drops_blank_rows_and_trims_whitespace(demo_raw_frame: pd.DataFrame) -> None:
    prepared, actions = prepare_frame(demo_raw_frame)
    counts = {action.name: action.count for action in actions}
    assert counts["blank_rows_dropped"] == 1
    assert counts["whitespace_normalised"] >= 2
    assert len(prepared) == len(demo_raw_frame) - 1
    assert prepared.at[46, "order_id"] == "SO-2025-9001"
    assert prepared.at[46, "product"] == "Desk Lamp"
    assert pd.isna(prepared.at[52, "quantity"])  # "N/A" becomes a missing value


def test_prepare_frame_handles_empty_input() -> None:
    prepared, actions = prepare_frame(pd.DataFrame())
    assert prepared.empty
    assert actions == ()


# ---------------------------------------------------------------------------
# Stage 2 - coercion
# ---------------------------------------------------------------------------
def test_coerce_types_converts_and_reports_failures(demo_raw_frame: pd.DataFrame) -> None:
    prepared, _ = prepare_frame(demo_raw_frame)
    coerced, failures = coerce_types(prepared)

    assert isinstance(coerced.at[2, "date"], pd.Timestamp)
    assert coerced.at[47, "unit_price"] == 1299.0
    assert coerced.at[49, "date"] is pd.NaT
    assert failures.dates[49] == "31/02/2025"
    assert failures.numeric["unit_price"] == {53: "abc"}
    assert failures.integers == {}


def test_coerce_types_upcasts_whole_numbers() -> None:
    frame = dataframe_from_values(
        [
            ["date", "order_id", "product", "quantity", "unit_price", "revenue", "region", "status"],
            ["2025-01-04", "A-1", "Widget", "2", "10", "20", "North", "Completed"],
        ]
    )
    prepared, _ = prepare_frame(frame)
    coerced, failures = coerce_types(prepared)
    assert failures.is_empty
    assert coerced.at[2, "quantity"] == 2
    assert coerced.at[2, "unit_price"] == 10.0
    assert coerced.at[2, "revenue"] == 20.0


# ---------------------------------------------------------------------------
# Stage 3 - cleaning
# ---------------------------------------------------------------------------
def test_clean_frame_produces_analytics_ready_data(quality) -> None:
    frame = quality.clean_data
    assert list(frame.columns)[:3] == [SOURCE_ROW_COLUMN, "date", "order_id"]
    assert ORDER_MONTH_COLUMN in frame.columns
    assert frame["order_id"].is_unique
    assert frame["date"].is_monotonic_increasing
    assert str(frame["quantity"].dtype) == "Int64"


def test_clean_frame_records_expected_actions(quality) -> None:
    actions = {action.name: action.count for action in quality.cleaning.actions}
    assert actions["rejected_rows_removed"] == quality.validation.rejected_rows
    assert actions["duplicate_orders_removed"] == 2
    assert actions["default_values_filled"] == 1
    assert actions["categorical_values_normalised"] >= 3
    assert actions["revenue_recalculated"] == 1


def test_clean_frame_fills_customer_default_and_recalculates_revenue(quality) -> None:
    frame = quality.clean_data
    defaulted = frame.loc[frame[SOURCE_ROW_COLUMN] == 55]
    assert defaulted.iloc[0]["customer"] == "Unknown Customer"

    recalculated = frame.loc[frame["order_id"] == "SO-2025-9003"].iloc[0]
    assert recalculated["revenue"] == 200.00
    assert recalculated["revenue"] == recalculated["quantity"] * recalculated["unit_price"]


def test_clean_frame_normalises_categoricals(quality) -> None:
    frame = quality.clean_data
    noisy = frame.loc[frame["order_id"] == "SO-2025-9001"].iloc[0]
    assert noisy["category"] == "Home & Kitchen"
    assert noisy["region"] == "North"
    assert noisy["status"] == "Completed"


def test_clean_frame_is_deterministic(quality, demo_raw_frame, validation_policy) -> None:
    from src.data.quality import run_quality_stages

    again = run_quality_stages(demo_raw_frame, policy=validation_policy)
    pd.testing.assert_frame_equal(quality.clean_data, again.clean_data)


def test_clean_frame_keeps_all_rows_when_nothing_was_rejected(
    demo_raw_frame: pd.DataFrame,
) -> None:
    from src.data.models import ValidationResult

    prepared, _ = prepare_frame(demo_raw_frame)
    coerced, _ = coerce_types(prepared)
    empty_validation = ValidationResult(
        total_rows=len(coerced), valid_rows=len(coerced), rejected=(), issues=()
    )
    result = clean_frame(coerced, empty_validation)
    assert result.rows_in == len(coerced)
    assert result.action_count("rejected_rows_removed") == 0
