"""Tests for the validation engine."""

from __future__ import annotations

import pytest

from src.data.cleaning import coerce_types, prepare_frame
from src.data.io import dataframe_from_values
from src.data.models import (
    CODE_DUPLICATE_ORDER,
    CODE_INVALID_CATEGORY,
    CODE_INVALID_DATE,
    CODE_INVALID_NUMBER,
    CODE_MISSING_VALUE,
    CODE_OUT_OF_RANGE,
    CODE_REVENUE_MISMATCH,
    CODE_REVENUE_OUTLIER,
)
from src.data.validation import ValidationPolicy, validate_frame
from src.utils.errors import ValidationError

MINIMAL_HEADERS = [
    "date",
    "order_id",
    "customer",
    "product",
    "category",
    "quantity",
    "unit_price",
    "revenue",
    "region",
    "status",
]

# Demo dataset: 58 rows, minus the blank row dropped during preparation = 57
# rows, of which 8 carry an error and are rejected.
EXPECTED_PREPARED_ROWS = 57
EXPECTED_REJECTED_ROWS = 8
EXPECTED_REJECTED_ROWS_BY_NUMBER = {49, 50, 51, 52, 53, 54, 56, 57}


def _row(**overrides: object) -> list[object]:
    base: dict[str, object] = {
        "date": "2025-01-05",
        "order_id": "T-1",
        "customer": "Test Customer",
        "product": "Widget",
        "category": "Electronics",
        "quantity": 2,
        "unit_price": 10.0,
        "revenue": 20.0,
        "region": "North",
        "status": "Completed",
    }
    base.update(overrides)
    return [base[header] for header in MINIMAL_HEADERS]


def _validate(rows: list[list[object]], **policy_kwargs: object):
    frame = dataframe_from_values([MINIMAL_HEADERS, *rows])
    prepared, _ = prepare_frame(frame)
    coerced, failures = coerce_types(prepared)
    return validate_frame(coerced, failures, ValidationPolicy(**policy_kwargs))


# ---------------------------------------------------------------------------
# Demo dataset behaviour (the documented defects)
# ---------------------------------------------------------------------------
def test_demo_dataset_validation_counts(quality) -> None:
    result = quality.validation
    assert result.total_rows == EXPECTED_PREPARED_ROWS
    assert result.rejected_rows == EXPECTED_REJECTED_ROWS
    assert result.valid_rows == EXPECTED_PREPARED_ROWS - EXPECTED_REJECTED_ROWS
    assert result.valid_rows + result.rejected_rows == result.total_rows


def test_every_error_code_from_the_demo_defects_is_reported(quality) -> None:
    codes = {issue.code for issue in quality.validation.issues if issue.severity == "error"}
    assert codes == {
        CODE_INVALID_DATE,
        CODE_INVALID_CATEGORY,
        CODE_INVALID_NUMBER,
        CODE_MISSING_VALUE,
        CODE_OUT_OF_RANGE,
    }


def test_warning_codes_from_the_demo_defects_are_reported(quality) -> None:
    codes = {issue.code for issue in quality.validation.issues if issue.severity == "warning"}
    assert {
        CODE_DUPLICATE_ORDER,
        CODE_REVENUE_MISMATCH,
        CODE_REVENUE_OUTLIER,
        CODE_MISSING_VALUE,
    } <= codes


def test_rejected_records_keep_their_row_number_and_reason(quality) -> None:
    rejected = {record.source_row: record for record in quality.validation.rejected}
    assert set(rejected) == EXPECTED_REJECTED_ROWS_BY_NUMBER

    bad_date = rejected[49]
    assert bad_date.order_id == "SO-2025-9004"
    assert CODE_INVALID_DATE in bad_date.codes
    assert "31/02/2025" in bad_date.reason_text()

    unknown_region = rejected[56]
    assert CODE_INVALID_CATEGORY in unknown_region.codes
    assert dict(unknown_region.values)["region"] == "Central"


def test_duplicates_are_warnings_not_rejections(quality) -> None:
    assert set(quality.validation.duplicate_order_ids) == {"SO-2025-0001", "SO-2025-0004"}
    assert quality.validation.duplicate_rows == 2
    issue = quality.validation.issues_for_code(CODE_DUPLICATE_ORDER)[0]
    assert issue.severity == "warning"


def test_outlier_row_is_flagged_but_kept(quality) -> None:
    assert quality.validation.outlier_rows == (58,)
    assert "SO-2025-9013" in set(quality.clean_data["order_id"])


def test_missing_customer_is_a_warning_with_default(quality) -> None:
    issues = [
        issue
        for issue in quality.validation.issues_for_code(CODE_MISSING_VALUE)
        if issue.column == "customer"
    ]
    assert issues and issues[0].severity == "warning"
    assert "Unknown Customer" in issues[0].message


def test_summary_lines_are_human_readable(quality) -> None:
    lines = quality.validation.summary_lines()
    assert any(line.startswith("rows read:") for line in lines)
    assert any(line.startswith("rows rejected:") for line in lines)


# ---------------------------------------------------------------------------
# Fatal conditions
# ---------------------------------------------------------------------------
def test_empty_dataset_is_fatal() -> None:
    frame = dataframe_from_values([MINIMAL_HEADERS])
    prepared, _ = prepare_frame(frame)
    coerced, failures = coerce_types(prepared)
    with pytest.raises(ValidationError, match="no data rows"):
        validate_frame(coerced, failures)


def test_missing_required_column_is_fatal() -> None:
    frame = dataframe_from_values([["date", "order_id"], ["2025-01-01", "T-1"]])
    prepared, _ = prepare_frame(frame)
    coerced, failures = coerce_types(prepared)
    with pytest.raises(ValidationError, match="required column"):
        validate_frame(coerced, failures)


# ---------------------------------------------------------------------------
# Rule level behaviour on minimal frames
# ---------------------------------------------------------------------------
def test_valid_record_produces_no_issues() -> None:
    result = _validate([_row()])
    assert result.issues == ()
    assert result.valid_rows == 1
    assert result.rejected == ()


def test_zero_quantity_is_rejected_as_out_of_range() -> None:
    result = _validate([_row(quantity=0, revenue=0.0)])
    assert result.rejected_rows == 1
    assert "minimum" in result.issues_for_code(CODE_OUT_OF_RANGE)[0].message


def test_non_numeric_price_is_rejected() -> None:
    result = _validate([_row(unit_price="abc")])
    assert result.rejected_rows == 1
    assert result.issues_for_code(CODE_INVALID_NUMBER)[0].column == "unit_price"


def test_blank_revenue_is_derived_with_a_warning() -> None:
    result = _validate([_row(revenue="")])
    assert result.rejected_rows == 0
    assert result.rows_with_code(CODE_MISSING_VALUE) == 1


def test_revenue_mismatch_is_a_warning() -> None:
    result = _validate([_row(revenue=999.0)])
    assert result.rejected_rows == 0
    assert result.rows_with_code(CODE_REVENUE_MISMATCH) == 1


def test_duplicate_order_ids_are_detected() -> None:
    result = _validate([_row(order_id="T-1"), _row(order_id="T-1")])
    assert result.duplicate_order_ids == ("T-1",)
    assert result.valid_rows == 2


def test_outlier_threshold_is_configurable() -> None:
    rows = [_row(order_id="T-1", quantity=10, unit_price=100.0, revenue=1000.0)]
    assert _validate(rows, revenue_outlier_threshold=500.0).outlier_rows == (2,)
    assert _validate(rows, revenue_outlier_threshold=10_000.0).outlier_rows == ()


def test_invalid_alias_category_is_still_accepted() -> None:
    result = _validate([_row(category="office supplies")])
    assert result.rejected_rows == 0
