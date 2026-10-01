from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.models import AIInsights, AIStatus, ClassificationRecord, ClassificationSummary
from src.analytics.kpi_engine import compute_analytics
from src.config import load_settings
from src.data.quality import QualityOutcome
from src.google_sheets.client import GoogleSheetsClient
from src.reporting.formatter import SheetFormatter
from src.reporting.models import CellFormat, SheetTable
from src.reporting.report_builder import (
    AI_GENERATED_NOTICE,
    RUN_LOG_HEADERS,
    build_ai_insights_table,
    build_clean_data_table,
    build_data_quality_table,
    build_dimension_table,
    build_feedback_table,
    build_kpi_summary_table,
    build_report_workbook,
    build_run_log_table,
)
from src.reporting.sink import GoogleSheetsSink, LocalReportSink
from tests.fixtures.fake_sheets_service import FakeSheetsService


@pytest.fixture
def analytics_bundle(quality: QualityOutcome):
    """Settings, quality stages and deterministic analytics for the demo data.

    Reuses the shared ``quality`` fixture (raw frame -> prepare -> coerce ->
    validate -> clean) so the reporting tests exercise the real pipeline.
    """
    settings = load_settings(env={})
    analytics = compute_analytics(quality.clean_data, source="unit_test")
    return {
        "settings": settings,
        "quality": quality,
        "cleaning": quality.cleaning,
        "analytics": analytics,
    }


def test_build_clean_data_table(analytics_bundle):
    cleaning = analytics_bundle["cleaning"]
    table = build_clean_data_table(cleaning.data, "Clean_Data")
    assert table.name == "Clean_Data"
    assert table.row_count == len(cleaning.data) + 1
    assert table.rows[0][0] == "source_row"
    assert "order_id" in table.rows[0]
    assert table.freeze_rows == 1
    assert len(table.number_formats) > 0


def test_build_data_quality_table(analytics_bundle):
    quality = analytics_bundle["quality"]
    blank_rows = sum(
        action.count
        for action in quality.preparation_actions
        if action.name == "blank_rows_dropped"
    )
    table = build_data_quality_table(
        quality.validation,
        quality.cleaning,
        "Data_Quality",
        prepared_rows=len(quality.prepared),
        blank_rows_dropped=blank_rows,
    )
    assert table.name == "Data_Quality"
    assert table.row_count > 10
    flat = [str(cell) for row in table.rows for cell in row]
    assert any("Data quality report" in cell for cell in flat)
    assert any("Validation findings" in cell for cell in flat)
    assert any("Cleaning actions" in cell for cell in flat)


def test_build_kpi_summary_table(analytics_bundle):
    analytics = analytics_bundle["analytics"]
    table = build_kpi_summary_table(analytics, "KPI_Summary")
    assert table.name == "KPI_Summary"
    flat = [str(cell) for row in table.rows for cell in row]
    assert any("Net revenue" in cell for cell in flat)
    assert any("Monthly trend" in cell for cell in flat)
    assert any("Order status distribution" in cell for cell in flat)


def test_build_dimension_table(analytics_bundle):
    analytics = analytics_bundle["analytics"]
    table = build_dimension_table(
        analytics.by_product,
        "By_Product",
        title="Product Performance (sorted by net revenue)",
        key_header="Product",
    )
    assert table.name == "By_Product"
    assert table.rows[0][0] == "Product Performance (sorted by net revenue)"
    assert table.rows[1][0] == "Product"
    assert len(table.rows) > 3


def test_build_ai_insights_table_disabled():
    insights = AIInsights.disabled("Feature toggle AI_ENABLED=false")
    table = build_ai_insights_table(insights, "AI_Insights")
    assert table.name == "AI_Insights"
    flat = " ".join(str(cell) for row in table.rows for cell in row)
    assert AI_GENERATED_NOTICE in flat
    assert "Feature toggle" in flat


def test_build_ai_insights_table_with_content():
    insights = AIInsights(
        status=AIStatus.GENERATED,
        status_detail="OK",
        provider_label="gemini:gemini-2.5-pro",
        model="gemini-2.5-pro",
        executive_summary="Solid sales across all regions.",
        positive_trends=("Electronics grew by 15%",),
        negative_trends=(),
        watch_items=("South region returns rising",),
        management_summary="Action: address return rates.",
        unverified_numbers=(999.99,),
        metrics_supplied=("net_revenue: 12000.0",),
    )
    table = build_ai_insights_table(insights, "AI_Insights")
    flat = " ".join(str(cell) for row in table.rows for cell in row)
    assert "Solid sales" in flat
    assert "Electronics grew by 15%" in flat
    assert "999.99" in flat


def test_build_feedback_table():
    summary = ClassificationSummary(
        status=AIStatus.GENERATED,
        status_detail="OK",
        provider_label="gemini:gemini-2.5-pro",
        model="gemini-2.5-pro",
        categories=("Pricing", "Delivery"),
        records=(
            ClassificationRecord(
                source_row=2, order_id="ORD-1", text="Fast shipping", category="Delivery"
            ),
        ),
        counts=(("Delivery", 1),),
    )
    table = build_feedback_table(summary, "Customer_Feedback")
    assert table.name == "Customer_Feedback"


def test_build_run_log_table():
    table = build_run_log_table(
        "Run_Log",
        run_id="run-123",
        started_at="2026-09-25T10:00:00Z",
        finished_at="2026-09-25T10:00:02Z",
        duration_seconds=2.04,
        outcome="success",
        data_source="demo",
        rows_read=58,
        rows_valid=55,
        rows_rejected=3,
        rows_cleaned=53,
        net_revenue=15420.50,
        ai_status="success",
        message="OK",
    )
    assert table.name == "Run_Log"
    assert table.rows[0] == RUN_LOG_HEADERS
    assert table.rows[1][0] == "run-123"
    assert table.rows[1][4] == "success"
    assert table.rows[1][10] == 15420.50


def test_build_report_workbook(analytics_bundle):
    wb = build_report_workbook(
        settings=analytics_bundle["settings"],
        cleaning=analytics_bundle["cleaning"],
        quality=analytics_bundle["quality"],
        analytics=analytics_bundle["analytics"],
        insights=AIInsights.disabled("test"),
        run_id="run-test",
        started_at="2026-09-25T10:00:00Z",
        finished_at="2026-09-25T10:00:01Z",
        duration_seconds=1.2,
        outcome="success",
        message="All good",
    )
    assert len(wb.report_tables) == 7
    assert wb.run_log_table is not None
    assert len(wb.all_tables) == 8


def test_local_report_sink(analytics_bundle, tmp_path: Path):
    wb = build_report_workbook(
        settings=analytics_bundle["settings"],
        cleaning=analytics_bundle["cleaning"],
        quality=analytics_bundle["quality"],
        analytics=analytics_bundle["analytics"],
        insights=AIInsights.disabled("test"),
        run_id="run-test",
        started_at="2026-09-25T10:00:00Z",
        finished_at="2026-09-25T10:00:01Z",
        duration_seconds=1.2,
        outcome="success",
        message="All good",
    )
    sink = LocalReportSink(tmp_path)
    result = sink.write(wb)

    assert result.table_count == 8
    assert (tmp_path / "KPI_Summary.csv").is_file()
    assert (tmp_path / "Data_Quality.csv").is_file()
    assert (tmp_path / "Clean_Data.csv").is_file()
    assert (tmp_path / "Run_Log.csv").is_file()
    category_csv = (tmp_path / "Category_Analysis.csv").read_text(encoding="utf-8")
    assert "Category" in category_csv


def test_formatting_requests_generation():
    table = SheetTable(
        name="Test",
        rows=(("H1", "H2"), (1, 2)),
        header_rows=(0,),
        freeze_rows=1,
        number_formats=(CellFormat(1, 0, "#,##0"),),
        column_widths=((0, 100),),
        wrap_columns=(1,),
    )
    requests = SheetFormatter.format_table(table, sheet_id=42)
    types = [list(req.keys())[0] for req in requests]
    assert "updateSheetProperties" in types
    assert "repeatCell" in types
    assert "updateDimensionProperties" in types


def test_google_sheets_sink_with_fake_service(analytics_bundle):
    fake = FakeSheetsService(worksheets={"Existing": []}, spreadsheet_id="test_sheet")
    client = GoogleSheetsClient(fake, "test_sheet")
    sink = GoogleSheetsSink(client, apply_formatting=True)

    wb = build_report_workbook(
        settings=analytics_bundle["settings"],
        cleaning=analytics_bundle["cleaning"],
        quality=analytics_bundle["quality"],
        analytics=analytics_bundle["analytics"],
        insights=AIInsights.disabled("test"),
        run_id="run-test",
        started_at="2026-09-25T10:00:00Z",
        finished_at="2026-09-25T10:00:01Z",
        duration_seconds=1.2,
        outcome="success",
        message="All good",
    )

    res = sink.write("test_sheet", wb)
    assert res.run_log_row_appended is True
    assert len(res.sheets_created) == 8
    assert res.format_requests_sent > 0

    res2 = sink.write("test_sheet", wb)
    assert res2.run_log_row_appended is True
    assert res2.sheets_created == ()

