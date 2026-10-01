"""Turn pipeline results into a workbook description.

The builder owns *presentation decisions for a business reader*: which sections a
sheet contains, in which order, and which cells are money/integer/percentage
values. It performs no calculations of its own - every number comes from the
deterministic analytics engine, the validation report or the cleaning report.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from src.ai.models import AI_GENERATED_NOTICE, AIInsights, ClassificationSummary
from src.analytics.models import AnalyticsResult, GroupMetric
from src.config.settings import Settings
from src.data.models import CleaningResult, ValidationResult
from src.data.quality import QualityOutcome
from src.data.cleaning import ORDER_MONTH_COLUMN, SOURCE_ROW_COLUMN
from src.data.io import values_from_dataframe
from src.reporting.models import (
    INTEGER_PATTERN,
    MONEY_PATTERN,
    PERCENT_PATTERN,
    CellFormat,
    SheetTable,
    WorkbookReport,
)

TEXT_CHUNK_SIZE = 500

RUN_LOG_HEADERS: tuple[str, ...] = (
    "run_id",
    "started_at",
    "finished_at",
    "duration_seconds",
    "outcome",
    "data_source",
    "rows_read",
    "rows_valid",
    "rows_rejected",
    "rows_cleaned",
    "net_revenue",
    "ai_status",
    "message",
)


class _TableBuilder:
    """Accumulates rows while remembering header rows, formats and widths."""

    def __init__(self, name: str, *, freeze_rows: int = 1) -> None:
        self.name = name
        self.freeze_rows = freeze_rows
        self._rows: list[tuple[Any, ...]] = []
        self._headers: list[int] = []
        self._formats: list[CellFormat] = []
        self._widths: list[tuple[int, int]] = []
        self._wrap: list[int] = []

    # -- rows -----------------------------------------------------------
    def line(
        self,
        values: Sequence[Any],
        *,
        header: bool = False,
        money: Iterable[int] = (),
        integer: Iterable[int] = (),
        percent: Iterable[int] = (),
    ) -> int:
        """Append a row and register per-cell number formats."""
        index = len(self._rows)
        self._rows.append(tuple(values))
        if header:
            self._headers.append(index)
        for column in money:
            self._formats.append(CellFormat(index, column, MONEY_PATTERN))
        for column in integer:
            self._formats.append(CellFormat(index, column, INTEGER_PATTERN))
        for column in percent:
            self._formats.append(CellFormat(index, column, PERCENT_PATTERN))
        return index

    def blank(self, columns: int = 1) -> None:
        self._rows.append(tuple("" for _ in range(max(columns, 1))))

    def paragraphs(self, text: str, *, columns: int = 1, column: int = 0) -> None:
        """Write long text as readable chunks (Sheets cells are not scrollable)."""
        for chunk in _chunk_text(text, TEXT_CHUNK_SIZE):
            row = [""] * max(columns, column + 1)
            row[column] = chunk
            self.line(row)

    def bullets(self, items: Iterable[str], *, columns: int = 1) -> None:
        for item in items:
            self.line([f"- {item}", *[""] * (max(columns, 1) - 1)])

    def width(self, column: int, pixels: int) -> None:
        self._widths.append((column, pixels))

    def wrap(self, *columns: int) -> None:
        for column in columns:
            if column not in self._wrap:
                self._wrap.append(column)

    # -- output ---------------------------------------------------------
    def build(self, *, minimum_columns: int = 1) -> SheetTable:
        rows = list(self._rows)
        if not rows:
            rows = [("",)]
        width = max([minimum_columns] + [len(row) for row in rows])
        padded = tuple(tuple(row) + ("",) * (width - len(row)) for row in rows)
        return SheetTable(
            name=self.name,
            rows=padded,
            header_rows=tuple(self._headers) or (0,),
            freeze_rows=self.freeze_rows,
            number_formats=tuple(self._formats),
            column_widths=tuple(self._widths),
            wrap_columns=tuple(self._wrap),
        )


def _chunk_text(text: str, size: int) -> list[str]:
    """Split long text on word boundaries into chunks of at most ``size`` chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Raw / cleaned data
# ---------------------------------------------------------------------------
def build_raw_data_table(frame: pd.DataFrame, worksheet: str) -> SheetTable:
    """The data exactly as it was read (original headers preserved)."""
    rows = values_from_dataframe(frame) if not frame.empty else [[""]]
    return SheetTable(
        name=worksheet,
        rows=tuple(tuple(row) for row in rows),
        header_rows=(0,),
        freeze_rows=1,
    )


def build_clean_data_table(frame: pd.DataFrame, worksheet: str) -> SheetTable:
    """The analytics-ready data, with money/integer columns formatted."""
    if frame.empty:
        return SheetTable(name=worksheet, rows=(tuple(frame.columns),), header_rows=(0,))

    rows = values_from_dataframe(frame)
    columns = list(frame.columns)
    ordered = [SOURCE_ROW_COLUMN, "date", "order_id", "customer", "product", "category",
               "quantity", "unit_price", "revenue", "region", "status", ORDER_MONTH_COLUMN,
               "feedback"]
    columns = [column for column in ordered if column in columns] + [
        column for column in columns if column not in ordered
    ]
    rows = values_from_dataframe(frame[columns])

    formats: list[CellFormat] = []
    for position, column in enumerate(columns):
        pattern = None
        if column in ("revenue", "unit_price"):
            pattern = MONEY_PATTERN
        elif column == "quantity":
            pattern = INTEGER_PATTERN
        if pattern is not None:
            for row_index in range(1, len(rows)):
                formats.append(CellFormat(row_index, position, pattern))

    return SheetTable(
        name=worksheet,
        rows=tuple(tuple(row) for row in rows),
        header_rows=(0,),
        freeze_rows=1,
        number_formats=tuple(formats),
        column_widths=((0, 90), (13, 320)),
    )


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def build_data_quality_table(
    validation: ValidationResult,
    cleaning: CleaningResult,
    worksheet: str,
    *,
    prepared_rows: int,
    blank_rows_dropped: int,
) -> SheetTable:
    """Validation findings, rejected records and cleaning actions - all visible."""
    builder = _TableBuilder(worksheet)
    builder.line(["Data quality report", ""], header=True)
    builder.line(["Rows read from the source", validation.total_rows + blank_rows_dropped], integer=(1,))
    builder.line(["Blank rows dropped before validation", blank_rows_dropped], integer=(1,))
    builder.line(["Rows validated", validation.total_rows], integer=(1,))
    builder.line(["Rows valid", validation.valid_rows], integer=(1,))
    builder.line(["Rows rejected", validation.rejected_rows], integer=(1,))
    builder.line(["Errors", validation.error_count], integer=(1,))
    builder.line(["Warnings", validation.warning_count], integer=(1,))
    builder.line(["Duplicate order ids found", len(validation.duplicate_order_ids)], integer=(1,))
    builder.line(["Revenue outliers flagged for review", len(validation.outlier_rows)], integer=(1,))
    builder.line(["Rows written to Clean_Data", cleaning.rows_out], integer=(1,))
    builder.line(["Rows removed by cleaning", cleaning.rows_removed], integer=(1,))
    builder.blank(2)

    builder.line(["Validation findings", ""], header=True)
    builder.line(["Code", "Severity", "Column", "Records", "Message", "Examples"], header=True)
    if validation.issues:
        for issue in validation.issues:
            builder.line(
                [
                    issue.code,
                    issue.severity,
                    issue.column or "",
                    issue.count,
                    issue.message,
                    "; ".join(issue.examples),
                ],
                integer=(3,),
            )
    else:
        builder.line(["none", "", "", 0, "no validation findings", ""], integer=(3,))
    builder.blank(6)

    builder.line(["Rejected records (kept for review - nothing is silently discarded)", ""], header=True)
    builder.line(["Source row", "Order id", "Codes", "Reason"], header=True)
    if validation.rejected:
        for record in validation.rejected:
            builder.line(
                [record.source_row, record.order_id or "", ", ".join(record.codes), record.reason_text()],
                integer=(0,),
            )
    else:
        builder.line(["", "", "", "no records were rejected"], integer=(0,))
    builder.blank(4)

    builder.line(["Cleaning actions", ""], header=True)
    builder.line(["Action", "Records", "Detail", "Examples"], header=True)
    if cleaning.actions:
        for action in cleaning.actions:
            builder.line(
                [action.name, action.count, action.detail, "; ".join(action.examples)],
                integer=(1,),
            )
    else:
        builder.line(["none", 0, "the dataset required no cleaning", ""], integer=(1,))

    builder.width(0, 260)
    builder.width(1, 100)
    builder.width(3, 90)
    builder.width(4, 420)
    builder.width(5, 420)
    builder.wrap(4, 5)
    return builder.build(minimum_columns=6)


# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------
def build_kpi_summary_table(analytics: AnalyticsResult, worksheet: str) -> SheetTable:
    """The business-facing KPI sheet: totals, period change, trend, status mix."""
    builder = _TableBuilder(worksheet)
    overall = analytics.overall

    builder.line(["KPI Summary", ""], header=True)
    builder.line(["Metric", "Value"], header=True)
    builder.line(["Reporting period analysed", analytics.period_label])
    builder.line(["Data source", analytics.source])
    builder.line(["Records analysed", overall.order_count], integer=(1,))
    builder.line(["Distinct customers", overall.distinct_customers], integer=(1,))
    builder.line(["Units sold", overall.total_quantity], integer=(1,))
    builder.line(["Gross revenue", overall.gross_revenue], money=(1,))
    builder.line(["Net revenue (excludes cancelled/refunded)", overall.net_revenue], money=(1,))
    builder.line(["Cancelled/refunded revenue", overall.excluded_revenue], money=(1,))
    builder.line(["Average order value (net)", overall.average_order_value], money=(1,))
    builder.line(["Average unit price (net)", overall.average_unit_price], money=(1,))
    builder.line(["Cancelled/refunded share of orders", overall.cancellation_rate_pct], percent=(1,))
    builder.blank(2)

    for snapshot, label in (
        (analytics.current_period, "Current period"),
        (analytics.previous_period, "Previous period"),
    ):
        if snapshot is None:
            continue
        builder.line([label, snapshot.label], header=True)
        builder.line(["Period start", snapshot.period_start])
        builder.line(["Period end", snapshot.period_end])
        builder.line(["Orders", snapshot.order_count], integer=(1,))
        builder.line(["Revenue bearing orders", snapshot.revenue_orders], integer=(1,))
        builder.line(["Units sold", snapshot.total_quantity], integer=(1,))
        builder.line(["Distinct customers", snapshot.distinct_customers], integer=(1,))
        builder.line(["Gross revenue", snapshot.gross_revenue], money=(1,))
        builder.line(["Net revenue", snapshot.net_revenue], money=(1,))
        builder.line(["Average order value (net)", snapshot.average_order_value], money=(1,))
        builder.blank(2)

    _add_comparison_section(builder, analytics)
    _add_trend_section(builder, analytics)
    _add_status_section(builder, analytics)

    builder.line(["Metric definitions (applied by the deterministic KPI engine)", ""], header=True)
    for definition in (
        "gross revenue: sum of revenue for every cleaned record in scope",
        "net revenue: revenue excluding orders with status Cancelled or Refunded",
        "average order value: net revenue / revenue bearing orders",
        "average unit price: net revenue / units in revenue bearing orders",
        "revenue share: share of the dataset's net revenue",
        "change %: period-over-period change; blank when the previous value is zero",
    ):
        builder.line([definition])
    builder.line(["AI interpretation is written to the AI_Insights worksheet", ""])
    builder.line(["Analysis generated at (UTC)", analytics.generated_at])
    builder.width(0, 380)
    builder.width(1, 160)
    return builder.build(minimum_columns=2)


def _add_comparison_section(builder: _TableBuilder, analytics: AnalyticsResult) -> None:
    builder.line(
        [f"Period-over-period change ({analytics.comparison_period_label})", ""], header=True
    )
    builder.line(["Metric", "Current", "Previous", "Change", "Change %"], header=True)
    if not analytics.comparisons:
        builder.line(["not available", "", "", "", ""])
        builder.line(
            [
                "A period comparison needs at least two distinct months of data; "
                "the dataset contains only one.",
            ]
        )
        builder.blank(5)
        return
    for comparison in analytics.comparisons:
        values = [
            comparison.metric,
            comparison.current,
            comparison.previous,
            comparison.change,
            "" if comparison.change_pct is None else comparison.change_pct,
        ]
        is_money = "revenue" in comparison.metric or "value" in comparison.metric
        if is_money:
            builder.line(values, money=(1, 2, 3), percent=(4,))
        else:
            builder.line(values, integer=(1, 2, 3), percent=(4,))
    builder.blank(5)


def _add_trend_section(builder: _TableBuilder, analytics: AnalyticsResult) -> None:
    builder.line(["Monthly trend", ""], header=True)
    builder.line(["Period", "Orders", "Net revenue"], header=True)
    if analytics.trend:
        for point in analytics.trend:
            builder.line(
                [point.label, point.order_count, point.net_revenue], integer=(1,), money=(2,)
            )
    else:
        builder.line(["no periods available", "", ""])
    builder.blank(3)


def _add_status_section(builder: _TableBuilder, analytics: AnalyticsResult) -> None:
    builder.line(["Order status distribution", ""], header=True)
    builder.line(["Status", "Orders", "Share of orders"], header=True)
    orders = analytics.overall.order_count
    if analytics.status_distribution:
        for status, count in analytics.status_distribution:
            share = round(100.0 * count / orders, 1) if orders else 0.0
            builder.line([status, count, share], integer=(1,), percent=(2,))
    else:
        builder.line(["no data available", "", ""])
    builder.blank(3)


# ---------------------------------------------------------------------------
# Dimension sheets
# ---------------------------------------------------------------------------
DIMENSION_HEADERS: tuple[str, ...] = (
    "Orders",
    "Units sold",
    "Gross revenue",
    "Net revenue",
    "Share of net revenue",
    "Average order value (net)",
)


def build_dimension_table(
    metrics: Sequence[GroupMetric],
    worksheet: str,
    *,
    title: str,
    key_header: str,
) -> SheetTable:
    """Generic product/category/region performance sheet."""
    columns = 1 + len(DIMENSION_HEADERS)
    builder = _TableBuilder(worksheet)
    builder.line([title, *[""] * (columns - 1)], header=True)
    builder.line([key_header, *DIMENSION_HEADERS], header=True)
    if metrics:
        for metric in metrics:
            builder.line(
                [
                    metric.key,
                    metric.order_count,
                    metric.total_quantity,
                    metric.gross_revenue,
                    metric.net_revenue,
                    metric.revenue_share_pct,
                    metric.average_order_value,
                ],
                integer=(1, 2),
                money=(3, 4, 6),
                percent=(5,),
            )
    else:
        builder.line(["no data available", *[""] * (columns - 1)])

    builder.blank(columns)
    builder.line(["Focus", *[""] * (columns - 1)], header=True)
    if metrics:
        builder.line([f"Highest net revenue: {metrics[0].key}", metrics[0].net_revenue], money=(1,))
        builder.line([f"Lowest net revenue: {metrics[-1].key}", metrics[-1].net_revenue], money=(1,))
        builder.line(
            ["Total net revenue", round(sum(metric.net_revenue for metric in metrics), 2)],
            money=(1,),
        )
    else:
        builder.line(["Not enough data to identify best or worst performers", ""])

    builder.width(0, 320)
    builder.width(1, 110)
    builder.width(3, 140)
    builder.width(4, 140)
    builder.width(5, 160)
    builder.width(6, 180)
    return builder.build(minimum_columns=columns)


# ---------------------------------------------------------------------------
# AI sheets
# ---------------------------------------------------------------------------
def build_ai_insights_table(insights: AIInsights, worksheet: str) -> SheetTable:
    """The AI interpretation sheet.

    Always written, even when AI is unavailable: the status block explains why,
    and the sheet clearly labels everything below it as AI-generated.
    """
    builder = _TableBuilder(worksheet)
    builder.line(["AI Insights (AI-generated interpretation layer)", ""], header=True)
    builder.line(["Status", insights.status.label()], header=True)
    builder.line(["Details", insights.status_detail])
    builder.line(["Provider", insights.provider_label or "n/a"])
    builder.line(["Model", insights.model or "n/a"])
    builder.line(["Generated at (UTC)", insights.generated_at or "n/a"])
    builder.line(["Important", AI_GENERATED_NOTICE])
    builder.line(
        [
            "Source of the numbers",
            "All figures in this workbook are calculated by the deterministic Python/pandas "
            "KPI engine. The AI only interprets them and never changes the data.",
        ]
    )
    builder.blank(2)

    if not insights.has_content:
        builder.line(["No AI narrative was produced for this run.", ""], header=True)
        builder.line([insights.status_detail])
        builder.width(0, 260)
        builder.width(1, 700)
        return builder.build(minimum_columns=2)

    sections: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("Executive summary", insights.executive_summary, ()),
        ("Positive trends", "", insights.positive_trends),
        ("Negative trends", "", insights.negative_trends),
        ("Items requiring attention", "", insights.watch_items),
        ("Management summary", insights.management_summary, ()),
    )
    for title, paragraph, bullets in sections:
        builder.line([title, ""], header=True)
        if paragraph:
            builder.paragraphs(paragraph, columns=2, column=0)
        for bullet in bullets:
            builder.line([f"- {bullet}", ""])
        builder.blank(2)

    builder.line(["Verification", ""], header=True)
    if insights.unverified_numbers:
        builder.line(
            [
                "Numbers not matched to a calculated metric",
                ", ".join(f"{value:g}" for value in insights.unverified_numbers),
            ]
        )
        builder.line(
            [
                "Note",
                "These figures could not be matched to a value supplied by the KPI engine. "
                "Treat them as unverified and check the source metric before using them.",
            ]
        )
    else:
        builder.line(["Number check", "no unverified figures detected in the AI narrative"])
    builder.line(
        [
            "Metrics supplied to the model",
            ", ".join(insights.metrics_supplied) if insights.metrics_supplied else "n/a",
        ]
    )
    builder.width(0, 260)
    builder.width(1, 700)
    builder.wrap(0, 1)
    return builder.build(minimum_columns=2)


def build_feedback_table(summary: ClassificationSummary, worksheet: str) -> SheetTable:
    """Optional AI text classification results (only written when available)."""
    builder = _TableBuilder(worksheet)
    builder.line(["Customer feedback classification (AI-generated)", ""], header=True)
    builder.line(["Status", summary.status.label()], header=True)
    builder.line(["Details", summary.status_detail])
    builder.line(["Provider", summary.provider_label or "n/a"])
    builder.line(["Model", summary.model or "n/a"])
    builder.line(["Categories", ", ".join(summary.categories) if summary.categories else "n/a"])
    builder.line(["Important", AI_GENERATED_NOTICE])
    builder.blank(2)

    builder.line(["Category distribution", ""], header=True)
    builder.line(["Category", "Records", "Share"], header=True)
    if summary.counts:
        for category, count in summary.counts:
            builder.line([category, count, summary.share_pct(category)], integer=(1,), percent=(2,))
    else:
        builder.line(["no classifications", 0, 0.0], integer=(1,), percent=(2,))
    if summary.unclassified:
        builder.line(
            ["unclassified (model returned no usable label)", summary.unclassified, ""], integer=(1,)
        )
    builder.blank(3)

    builder.line(["Classified records", ""], header=True)
    builder.line(["Source row", "Order id", "Category (AI)", "Feedback text"], header=True)
    if summary.records:
        for record in summary.records:
            builder.line(
                [record.source_row, record.order_id, record.category, record.text], integer=(0,)
            )
    else:
        builder.line(["", "", "", "no records were classified"])
    builder.width(0, 110)
    builder.width(1, 140)
    builder.width(2, 180)
    builder.width(3, 620)
    builder.wrap(3)
    return builder.build(minimum_columns=4)


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------
def build_run_log_table(
    worksheet: str,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    outcome: str,
    data_source: str,
    rows_read: int,
    rows_valid: int,
    rows_rejected: int,
    rows_cleaned: int,
    net_revenue: float | None,
    ai_status: str,
    message: str,
) -> SheetTable:
    """One row of run history (appended, never overwritten)."""
    row: tuple[Any, ...] = (
        run_id,
        started_at,
        finished_at,
        round(duration_seconds, 2),
        outcome,
        data_source,
        rows_read,
        rows_valid,
        rows_rejected,
        rows_cleaned,
        "" if net_revenue is None else net_revenue,
        ai_status,
        message,
    )
    return SheetTable(
        name=worksheet,
        rows=(RUN_LOG_HEADERS, row),
        header_rows=(0,),
        freeze_rows=1,
        number_formats=(
            CellFormat(1, 3, "0.00"),
            CellFormat(1, 6, INTEGER_PATTERN),
            CellFormat(1, 7, INTEGER_PATTERN),
            CellFormat(1, 8, INTEGER_PATTERN),
            CellFormat(1, 9, INTEGER_PATTERN),
            CellFormat(1, 10, MONEY_PATTERN),
        ),
        column_widths=((12, 420),),
    )



def build_report_workbook(
    settings: Settings,
    cleaning: CleaningResult,
    quality: QualityOutcome,
    analytics: AnalyticsResult,
    insights: AIInsights,
    run_id: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    outcome: str,
    message: str,
) -> WorkbookReport:
    """Build a complete report workbook from pipeline results.

    Worksheet titles come from the configured worksheet names, so renaming a
    tab in the environment configuration propagates to every table. The Run_Log
    table goes into ``append_tables``: sinks append it so run history grows.
    """
    sheets = settings.google_sheets.worksheets
    blank_rows_dropped = sum(
        action.count
        for action in quality.preparation_actions
        if action.name == "blank_rows_dropped"
    )

    report_tables = (
        build_clean_data_table(cleaning.data, sheets.clean_data),
        build_data_quality_table(
            quality.validation,
            quality.cleaning,
            sheets.data_quality,
            prepared_rows=len(quality.prepared),
            blank_rows_dropped=blank_rows_dropped,
        ),
        build_kpi_summary_table(analytics, sheets.kpi_summary),
        build_dimension_table(
            analytics.by_product,
            sheets.product_analysis,
            title="Product Performance (sorted by net revenue)",
            key_header="Product",
        ),
        build_dimension_table(
            analytics.by_category,
            sheets.category_analysis,
            title="Category Performance (sorted by net revenue)",
            key_header="Category",
        ),
        build_dimension_table(
            analytics.by_region,
            sheets.regional_analysis,
            title="Regional Performance (sorted by net revenue)",
            key_header="Region",
        ),
        build_ai_insights_table(insights, sheets.ai_insights),
    )

    run_log_table = build_run_log_table(
        sheets.run_log,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        outcome=outcome,
        data_source="pipeline",
        rows_read=quality.validation.total_rows + blank_rows_dropped,
        rows_valid=quality.validation.valid_rows,
        rows_rejected=quality.validation.rejected_rows,
        rows_cleaned=cleaning.rows_out,
        net_revenue=analytics.overall.net_revenue,
        ai_status=insights.status.label(),
        message=message,
    )

    return WorkbookReport(tables=report_tables, append_tables=(run_log_table,))
