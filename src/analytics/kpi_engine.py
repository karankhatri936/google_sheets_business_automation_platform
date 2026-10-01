"""Deterministic business KPI engine.

Pure pandas: this module is the single source of truth for every number the
application reports. The AI layer only ever *interprets* what is computed here.

Documented metric rules (see also ``docs/BUSINESS_RULES.md``):

* ``gross_revenue``  - sum of revenue across all records in scope.
* ``net_revenue``    - sum of revenue excluding :data:`NON_REVENUE_STATUSES`
  (Cancelled / Refunded), i.e. revenue that was actually realised.
* ``average_order_value`` - net_revenue / number of revenue bearing orders.
* ``average_unit_price``  - net_revenue / units in revenue bearing orders.
* ``revenue_share_pct``   - share of the dataset's net revenue, rounded to 0.1%.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.analytics.models import (
    AnalyticsResult,
    GroupMetric,
    KpiSnapshot,
    PeriodChange,
    TrendPoint,
)
from src.config.logging_config import get_logger
from src.config.schema import NON_REVENUE_STATUSES
from src.utils.errors import ValidationError

REQUIRED_ANALYTICS_COLUMNS: tuple[str, ...] = (
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
    "order_month",
)

DIMENSION_COLUMNS: tuple[str, ...] = ("product", "category", "region")
_PERIOD_METRICS: tuple[tuple[str, str], ...] = (
    ("net_revenue", "net revenue"),
    ("gross_revenue", "gross revenue"),
    ("order_count", "orders"),
    ("average_order_value", "average order value"),
    ("total_quantity", "units sold"),
    ("distinct_customers", "distinct customers"),
)


def _revenue_bearing(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose status represents realised revenue."""
    return frame.loc[~frame["status"].isin(NON_REVENUE_STATUSES)]


def _money(value: float) -> float:
    return round(float(value), 2)


def _snapshot(frame: pd.DataFrame, label: str) -> KpiSnapshot:
    """Headline KPIs for one scope of the cleaned dataset."""
    if frame.empty:
        return KpiSnapshot(
            label=label,
            period_start="",
            period_end="",
            order_count=0,
            revenue_orders=0,
            distinct_customers=0,
            total_quantity=0,
            revenue_quantity=0,
            gross_revenue=0.0,
            net_revenue=0.0,
            excluded_revenue=0.0,
            average_order_value=0.0,
            average_unit_price=0.0,
            status_counts=(),
        )

    revenue_rows = _revenue_bearing(frame)
    gross_revenue = _money(frame["revenue"].sum())
    net_revenue = _money(revenue_rows["revenue"].sum())
    revenue_orders = int(len(revenue_rows))
    revenue_quantity = int(revenue_rows["quantity"].sum()) if revenue_orders else 0
    counter = Counter(str(status) for status in frame["status"])
    status_counts = tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))

    return KpiSnapshot(
        label=label,
        period_start=frame["date"].min().strftime("%Y-%m-%d"),
        period_end=frame["date"].max().strftime("%Y-%m-%d"),
        order_count=int(len(frame)),
        revenue_orders=revenue_orders,
        distinct_customers=int(frame["customer"].nunique()),
        total_quantity=int(frame["quantity"].sum()),
        revenue_quantity=revenue_quantity,
        gross_revenue=gross_revenue,
        net_revenue=net_revenue,
        excluded_revenue=_money(gross_revenue - net_revenue),
        average_order_value=_money(net_revenue / revenue_orders) if revenue_orders else 0.0,
        average_unit_price=_money(net_revenue / revenue_quantity) if revenue_quantity else 0.0,
        status_counts=status_counts,
    )


def _group_metrics(
    frame: pd.DataFrame,
    column: str,
    *,
    total_net_revenue: float,
) -> tuple[GroupMetric, ...]:
    """Aggregate one dimension, ordered by net revenue (descending)."""
    metrics: list[GroupMetric] = []
    for key, group in frame.groupby(column, sort=True, dropna=False):
        revenue_rows = _revenue_bearing(group)
        net_revenue = _money(revenue_rows["revenue"].sum())
        revenue_orders = int(len(revenue_rows))
        metrics.append(
            GroupMetric(
                key=str(key),
                order_count=int(len(group)),
                total_quantity=int(group["quantity"].sum()),
                gross_revenue=_money(group["revenue"].sum()),
                net_revenue=net_revenue,
                revenue_share_pct=(
                    round(100.0 * net_revenue / total_net_revenue, 1) if total_net_revenue else 0.0
                ),
                average_order_value=(
                    _money(net_revenue / revenue_orders) if revenue_orders else 0.0
                ),
            )
        )
    metrics.sort(key=lambda metric: (-metric.net_revenue, metric.key))
    return tuple(metrics)


def _compare_periods(current: KpiSnapshot, previous: KpiSnapshot) -> tuple[PeriodChange, ...]:
    """Build the period-over-period comparison table."""
    changes: list[PeriodChange] = []
    for attribute, label in _PERIOD_METRICS:
        current_value = float(getattr(current, attribute))
        previous_value = float(getattr(previous, attribute))
        change = round(current_value - previous_value, 2)
        change_pct = round(100.0 * change / previous_value, 1) if previous_value else None
        changes.append(
            PeriodChange(
                metric=label,
                current=current_value,
                previous=previous_value,
                change=change,
                change_pct=change_pct,
            )
        )
    return tuple(changes)


def _period_labels(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the (current, previous) period labels from the available months."""
    months = sorted(str(month) for month in frame["order_month"].dropna().unique())
    if not months:
        return None, None
    current = months[-1]
    previous = months[-2] if len(months) > 1 else None
    return current, previous


def compute_analytics(
    clean_data: pd.DataFrame,
    *,
    top_n: int = 5,
    source: str = "unknown",
    generated_at: datetime | None = None,
) -> AnalyticsResult:
    """Compute every deterministic KPI from the cleaned dataset.

    Raises
    ------
    ValidationError
        When the dataset is empty or lacks the columns the engine needs.
    """
    logger = get_logger("analytics")
    if clean_data is None or clean_data.empty:
        raise ValidationError("cannot compute analytics on an empty dataset")

    missing = [column for column in REQUIRED_ANALYTICS_COLUMNS if column not in clean_data.columns]
    if missing:
        raise ValidationError(
            "cleaned dataset is missing column(s) required for analytics: " + ", ".join(missing)
        )
    if top_n < 1:
        raise ValidationError("top_n must be >= 1")

    overall = _snapshot(clean_data, "All periods")
    by_dimension = {
        column: _group_metrics(clean_data, column, total_net_revenue=overall.net_revenue)
        for column in DIMENSION_COLUMNS
    }

    current_label, previous_label = _period_labels(clean_data)
    current_period = (
        _snapshot(clean_data.loc[clean_data["order_month"] == current_label], str(current_label))
        if current_label is not None
        else None
    )
    previous_period = (
        _snapshot(clean_data.loc[clean_data["order_month"] == previous_label], str(previous_label))
        if previous_label is not None
        else None
    )
    comparisons = (
        _compare_periods(current_period, previous_period)
        if current_period is not None and previous_period is not None
        else ()
    )

    months = sorted(str(month) for month in clean_data["order_month"].dropna().unique())
    trend = tuple(
        TrendPoint(
            label=month,
            order_count=int((clean_data["order_month"] == month).sum()),
            net_revenue=_money(
                _revenue_bearing(clean_data.loc[clean_data["order_month"] == month])["revenue"].sum()
            ),
        )
        for month in months
    )

    by_product = by_dimension["product"]
    result = AnalyticsResult(
        overall=overall,
        by_product=by_product,
        by_category=by_dimension["category"],
        by_region=by_dimension["region"],
        trend=trend,
        top_products=by_product[:top_n],
        bottom_products=tuple(reversed(by_product[-top_n:])),
        current_period=current_period,
        previous_period=previous_period,
        comparisons=comparisons,
        generated_at=(generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=source,
        top_n=top_n,
    )
    logger.info(
        "analytics complete: %d order(s), net revenue %.2f, %d product(s), %d period(s)",
        overall.order_count,
        overall.net_revenue,
        len(by_product),
        len(trend),
    )
    return result


def analytics_is_available(clean_data: pd.DataFrame) -> tuple[bool, str]:
    """Cheap pre-flight check used by the CLI for precise error messages."""
    if clean_data is None or clean_data.empty:
        return False, "no cleaned rows are available"
    missing = [
        column for column in REQUIRED_ANALYTICS_COLUMNS if column not in clean_data.columns
    ]
    if missing:
        return False, "missing column(s): " + ", ".join(missing)
    return True, "ok"
