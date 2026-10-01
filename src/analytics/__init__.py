"""Deterministic analytics: KPI engine and its typed result models."""

from src.analytics.kpi_engine import (
    DIMENSION_COLUMNS,
    REQUIRED_ANALYTICS_COLUMNS,
    analytics_is_available,
    compute_analytics,
)
from src.analytics.models import (
    AnalyticsResult,
    GroupMetric,
    KpiSnapshot,
    PeriodChange,
    TrendPoint,
)

__all__ = [
    "DIMENSION_COLUMNS",
    "REQUIRED_ANALYTICS_COLUMNS",
    "AnalyticsResult",
    "GroupMetric",
    "KpiSnapshot",
    "PeriodChange",
    "TrendPoint",
    "analytics_is_available",
    "compute_analytics",
]
