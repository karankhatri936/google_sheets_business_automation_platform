"""Tests for the deterministic KPI engine.

The engine is cross-checked against an independent pandas calculation of the same
values (a different code path), so a regression in the engine cannot be masked by
matching the engine's own arithmetic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.analytics.kpi_engine import compute_analytics
from src.analytics.models import AnalyticsResult
from src.config.schema import NON_REVENUE_STATUSES
from src.utils.errors import ValidationError

GENERATED_AT = datetime(2025, 3, 1, tzinfo=timezone.utc)


@pytest.fixture
def analytics(clean_data: pd.DataFrame) -> AnalyticsResult:
    return compute_analytics(clean_data, top_n=5, source="demo", generated_at=GENERATED_AT)


def _net_mask(frame: pd.DataFrame) -> pd.Series:
    return ~frame["status"].isin(NON_REVENUE_STATUSES)


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------
def test_gross_and_net_revenue_match_independent_calculation(
    analytics: AnalyticsResult, clean_data: pd.DataFrame
) -> None:
    expected_gross = round(float(clean_data["revenue"].sum()), 2)
    expected_net = round(float(clean_data.loc[_net_mask(clean_data), "revenue"].sum()), 2)

    assert analytics.overall.gross_revenue == expected_gross
    assert analytics.overall.net_revenue == expected_net
    assert analytics.overall.excluded_revenue == round(expected_gross - expected_net, 2)
    assert analytics.overall.net_revenue <= analytics.overall.gross_revenue


def test_counts_and_averages_are_consistent(
    analytics: AnalyticsResult, clean_data: pd.DataFrame
) -> None:
    overall = analytics.overall
    assert overall.order_count == len(clean_data)
    assert overall.total_quantity == int(clean_data["quantity"].sum())
    assert overall.distinct_customers == int(clean_data["customer"].nunique())

    revenue_rows = clean_data.loc[_net_mask(clean_data)]
    assert overall.revenue_orders == len(revenue_rows)
    assert overall.average_order_value == round(overall.net_revenue / len(revenue_rows), 2)
    assert overall.average_unit_price == round(
        overall.net_revenue / int(revenue_rows["quantity"].sum()), 2
    )


def test_status_distribution_covers_every_order(analytics: AnalyticsResult) -> None:
    statuses = dict(analytics.status_distribution)
    assert sum(statuses.values()) == analytics.overall.order_count
    assert set(statuses) <= {"Completed", "Pending", "Cancelled", "Refunded"}
    assert analytics.overall.cancellation_rate_pct == round(
        100.0
        * (analytics.overall.order_count - analytics.overall.revenue_orders)
        / analytics.overall.order_count,
        1,
    )


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
def test_group_metrics_partition_the_net_revenue(analytics: AnalyticsResult) -> None:
    for dimension in (analytics.by_product, analytics.by_category, analytics.by_region):
        assert dimension
        assert round(sum(metric.net_revenue for metric in dimension), 2) == pytest.approx(
            analytics.overall.net_revenue, abs=0.05
        )
        assert round(sum(metric.revenue_share_pct for metric in dimension), 1) == pytest.approx(
            100.0, abs=0.6
        )


def test_group_metrics_are_sorted_by_net_revenue(analytics: AnalyticsResult) -> None:
    for dimension in (analytics.by_product, analytics.by_category, analytics.by_region):
        revenues = [metric.net_revenue for metric in dimension]
        assert revenues == sorted(revenues, reverse=True)
        keys = [metric.key for metric in dimension]
        assert len(keys) == len(set(keys))


def test_top_and_bottom_products(analytics: AnalyticsResult) -> None:
    assert analytics.top_products == analytics.by_product[:5]
    assert analytics.bottom_products == tuple(reversed(analytics.by_product[-5:]))
    top_revenues = [metric.net_revenue for metric in analytics.top_products]
    bottom_revenues = [metric.net_revenue for metric in analytics.bottom_products]
    assert top_revenues == sorted(top_revenues, reverse=True)
    assert bottom_revenues == sorted(bottom_revenues)


def test_top_n_is_configurable(clean_data: pd.DataFrame) -> None:
    result = compute_analytics(clean_data, top_n=2, generated_at=GENERATED_AT)
    assert len(result.top_products) == 2
    assert len(result.bottom_products) == 2


# ---------------------------------------------------------------------------
# Period analysis
# ---------------------------------------------------------------------------
def test_latest_and_previous_period_are_selected(analytics: AnalyticsResult) -> None:
    assert analytics.current_period is not None
    assert analytics.previous_period is not None
    assert analytics.current_period.label == "2025-02"
    assert analytics.previous_period.label == "2025-01"
    assert analytics.period_label == "2025-01 to 2025-02"
    assert analytics.comparison_period_label == "2025-02 vs 2025-01"


def test_period_snapshots_cover_the_whole_dataset(analytics: AnalyticsResult) -> None:
    current, previous = analytics.current_period, analytics.previous_period
    assert current is not None and previous is not None
    assert current.order_count + previous.order_count == analytics.overall.order_count
    assert current.net_revenue + previous.net_revenue == pytest.approx(
        analytics.overall.net_revenue, abs=0.02
    )


def test_comparisons_are_internally_consistent(analytics: AnalyticsResult) -> None:
    assert analytics.comparisons
    for comparison in analytics.comparisons:
        assert comparison.change == pytest.approx(comparison.current - comparison.previous, abs=0.01)
        if comparison.previous:
            assert comparison.change_pct == round(
                100.0 * comparison.change / comparison.previous, 1
            )
        else:
            assert comparison.change_pct is None
    labels = [comparison.metric for comparison in analytics.comparisons]
    assert "net revenue" in labels and "orders" in labels


def test_trend_series_matches_periods(analytics: AnalyticsResult) -> None:
    assert [point.label for point in analytics.trend] == ["2025-01", "2025-02"]
    assert sum(point.order_count for point in analytics.trend) == analytics.overall.order_count
    assert sum(point.net_revenue for point in analytics.trend) == pytest.approx(
        analytics.overall.net_revenue, abs=0.02
    )


def test_single_period_dataset_has_no_comparison(clean_data: pd.DataFrame) -> None:
    latest = clean_data["order_month"].max()
    assert isinstance(latest, str)
    single_period = clean_data.loc[clean_data["order_month"] == latest]
    result = compute_analytics(single_period, generated_at=GENERATED_AT)
    assert result.current_period is not None
    assert result.previous_period is None
    assert result.comparisons == ()
    assert result.period_label == latest


# ---------------------------------------------------------------------------
# Contract / determinism
# ---------------------------------------------------------------------------
def test_result_is_deterministic(clean_data: pd.DataFrame) -> None:
    first = compute_analytics(clean_data, source="demo", generated_at=GENERATED_AT)
    second = compute_analytics(clean_data, source="demo", generated_at=GENERATED_AT)
    assert first == second


def test_numeric_values_are_plain_python_types(analytics: AnalyticsResult) -> None:
    assert type(analytics.overall.net_revenue) is float
    assert type(analytics.overall.order_count) is int
    assert type(analytics.by_product[0].revenue_share_pct) is float


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValidationError, match="empty dataset"):
        compute_analytics(pd.DataFrame())


def test_missing_columns_are_rejected(clean_data: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="missing column"):
        compute_analytics(clean_data.drop(columns=["revenue"]))
