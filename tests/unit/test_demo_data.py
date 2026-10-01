"""Tests for the synthetic demo dataset."""

from __future__ import annotations

from src.data.demo_data import DEMO_DATA_DISCLAIMER, RAW_HEADERS, build_demo_dataset

# Documented row counts: 22 + 20 clean orders plus 16 deliberate defect rows.
EXPECTED_CLEAN_ROWS = 42
EXPECTED_DEFECT_ROWS = 16
EXPECTED_TOTAL_ROWS = EXPECTED_CLEAN_ROWS + EXPECTED_DEFECT_ROWS

DEFECT_ORDER_IDS = (
    "SO-2025-9001",
    "SO-2025-9002",
    "SO-2025-9003",
    "SO-2025-9004",
    "SO-2025-9005",
    "SO-2025-9006",
    "SO-2025-9007",
    "SO-2025-9008",
    "SO-2025-9009",
    "SO-2025-9010",
    "SO-2025-9011",
    "SO-2025-9012",
    "SO-2025-9013",
)


def test_dataset_is_deterministic() -> None:
    assert build_demo_dataset().rows == build_demo_dataset().rows


def test_row_counts_match_documented_shape(demo_dataset) -> None:
    assert demo_dataset.row_count == EXPECTED_TOTAL_ROWS
    assert len(demo_dataset.headers) == len(RAW_HEADERS)


def test_revenue_is_always_quantity_times_unit_price(demo_dataset) -> None:
    """Clean rows must derive revenue, never invent it."""
    for row in demo_dataset.rows[:EXPECTED_CLEAN_ROWS]:
        expected = round(float(row["Qty"]) * float(row["Unit Price"]), 2)
        assert row["Revenue"] == expected, row["Order ID"]


def test_defect_rows_cover_the_documented_rules(demo_dataset) -> None:
    # Values are compared stripped: one defect row deliberately carries
    # surrounding whitespace (it is normalised during cleaning).
    order_ids = [str(row["Order ID"]).strip() for row in demo_dataset.rows]
    assert order_ids.count("SO-2025-0001") == 2, "exact duplicate defect missing"
    assert order_ids.count("SO-2025-0004") == 2, "conflicting duplicate defect missing"
    for expected_id in DEFECT_ORDER_IDS:
        assert expected_id in order_ids, f"defect row {expected_id} missing"


def test_whitespace_defect_row_is_present(demo_dataset) -> None:
    noisy_ids = [
        str(row["Order ID"]) for row in demo_dataset.rows if str(row["Order ID"]) != str(row["Order ID"]).strip()
    ]
    assert noisy_ids == ["  SO-2025-9001  "]


def test_blank_row_exists(demo_dataset) -> None:
    blank_rows = [
        row
        for row in demo_dataset.rows
        if all(str(value).strip() == "" for value in row.values())
    ]
    assert len(blank_rows) == 1


def test_value_grid_shape(demo_dataset) -> None:
    grid = demo_dataset.to_values()
    assert grid[0] == list(RAW_HEADERS)
    assert len(grid) == demo_dataset.row_count + 1
    assert all(len(row) == len(RAW_HEADERS) for row in grid)


def test_csv_export_has_header_and_quotes_commas(demo_dataset) -> None:
    text = demo_dataset.to_csv_text()
    lines = text.strip().splitlines()
    assert lines[0].startswith("Order Date,")
    assert len(lines) == demo_dataset.row_count + 1
    assert '"' in text, "values containing commas must be quoted"


def test_disclaimer_labels_data_as_synthetic() -> None:
    assert "Synthetic" in DEMO_DATA_DISCLAIMER
