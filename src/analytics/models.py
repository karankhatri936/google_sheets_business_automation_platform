"""Structured results of the deterministic analytics engine.

The dataclasses here are the contract between:

* the KPI engine (produces them from cleaned pandas data),
* the reporting layer (writes them into the workbook),
* the AI interpretation layer (receives them as verified facts), and
* the tests (assert exact, reproducible numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KpiSnapshot:
    """Headline numbers for one scope (a period or the whole dataset)."""

    label: str
    period_start: str
    period_end: str
    order_count: int
    revenue_orders: int
    distinct_customers: int
    total_quantity: int
    revenue_quantity: int
    gross_revenue: float
    net_revenue: float
    excluded_revenue: float
    average_order_value: float
    average_unit_price: float
    status_counts: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def cancellation_rate_pct(self) -> float:
        """Share of orders that were cancelled or refunded."""
        if not self.order_count:
            return 0.0
        excluded = self.order_count - self.revenue_orders
        return round(100.0 * excluded / self.order_count, 1)


@dataclass(frozen=True)
class GroupMetric:
    """Aggregated performance for one dimension value (product, category, region)."""

    key: str
    order_count: int
    total_quantity: int
    gross_revenue: float
    net_revenue: float
    revenue_share_pct: float
    average_order_value: float


@dataclass(frozen=True)
class PeriodChange:
    """Period-over-period comparison of a single metric."""

    metric: str
    current: float
    previous: float
    change: float
    change_pct: float | None  # None when the previous value is zero


@dataclass(frozen=True)
class TrendPoint:
    """One point of the period trend series."""

    label: str
    order_count: int
    net_revenue: float


@dataclass(frozen=True)
class AnalyticsResult:
    """Complete deterministic analytics output for one dataset."""

    overall: KpiSnapshot
    by_product: tuple[GroupMetric, ...]
    by_category: tuple[GroupMetric, ...]
    by_region: tuple[GroupMetric, ...]
    trend: tuple[TrendPoint, ...]
    top_products: tuple[GroupMetric, ...]
    bottom_products: tuple[GroupMetric, ...]
    current_period: KpiSnapshot | None = None
    previous_period: KpiSnapshot | None = None
    comparisons: tuple[PeriodChange, ...] = ()
    generated_at: str = ""
    source: str = "unknown"
    top_n: int = 5

    @property
    def period_label(self) -> str:
        """Human readable label for the analysis window."""
        if self.current_period is None:
            return self.overall.label
        if self.previous_period is None:
            return self.current_period.label
        return f"{self.previous_period.label} to {self.current_period.label}"

    @property
    def comparison_period_label(self) -> str:
        """Label used for period-over-period statements."""
        if self.current_period is None or self.previous_period is None:
            return self.period_label
        return f"{self.current_period.label} vs {self.previous_period.label}"

    @property
    def status_distribution(self) -> tuple[tuple[str, int], ...]:
        return self.overall.status_counts
