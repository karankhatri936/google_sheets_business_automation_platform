"""End-to-end pipeline tests.

Fully offline: reads use the synthetic demo dataset or a ``FakeSheetsService``,
and AI uses the deterministic mock provider - no credentials, no network.
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pytest

from src.config import load_settings
from src.config.settings import SchedulerSettings
from src.data.demo_data import build_demo_dataset
from src.pipeline import build_trigger, run_pipeline
from src.pipeline.pipeline import OUTCOME_FAILED, PipelineResult
from src.utils.errors import ConfigurationError
from tests.fixtures.fake_sheets_service import FakeSheetsService

MOCK_AI = {"AI_ENABLED": "true", "AI_PROVIDER": "mock"}


def test_dry_run_pipeline_writes_local_workbook(tmp_path: Path):
    settings = load_settings(env={**MOCK_AI, "LOCAL_OUTPUT_DIR": str(tmp_path)})
    result = run_pipeline(settings, dry_run=True, use_demo_data=True)

    assert result.succeeded, result.message
    assert result.mode == "dry_run"
    assert result.data_source == "demo"
    assert result.rows_read > 10
    assert result.rows_cleaned > 0
    assert result.net_revenue is not None and result.net_revenue > 0
    assert result.insights is not None and result.insights.status.value == "generated"
    assert result.feedback is not None and result.feedback.status.value == "generated"

    for name in ("Clean_Data", "Data_Quality", "KPI_Summary", "AI_Insights", "Run_Log"):
        assert (tmp_path / f"{name}.csv").is_file()
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["outcome"] == "success"
    assert summary["run_id"] == result.run_id
    assert len(summary["tables"]) == 8


def test_live_pipeline_against_fake_sheets():
    demo_values = build_demo_dataset().to_values()
    fake = FakeSheetsService(
        worksheets={"Raw_Data": demo_values}, spreadsheet_id="sheet-live"
    )
    settings = load_settings(env={**MOCK_AI, "GS_SPREADSHEET_ID": "sheet-live"})
    result = run_pipeline(settings, sheets_service=fake)

    assert result.succeeded, result.message
    assert result.mode == "live"
    assert result.data_source == "google_sheets"
    expected = {
        "Raw_Data",
        "Clean_Data",
        "Data_Quality",
        "KPI_Summary",
        "Product_Analysis",
        "Category_Analysis",
        "Regional_Analysis",
        "AI_Insights",
        "Run_Log",
    }
    assert expected <= set(fake.worksheets)
    run_log = fake.read("Run_Log")
    assert run_log[0][0] == "run_id"
    assert len(run_log) == 2  # header row + one appended run row
    assert len(fake.read("Clean_Data")) == result.rows_cleaned + 1


def test_live_pipeline_trims_run_log_to_configured_max():
    demo_values = build_demo_dataset().to_values()
    seeded_log = [[f"old-{index}"] for index in range(15)]
    fake = FakeSheetsService(
        worksheets={"Raw_Data": demo_values, "Run_Log": seeded_log},
        spreadsheet_id="sheet-trim",
    )
    settings = load_settings(
        env={
            **MOCK_AI,
            "GS_SPREADSHEET_ID": "sheet-trim",
            "GS_RUN_LOG_MAX_ROWS": "10",
        }
    )
    result = run_pipeline(settings, sheets_service=fake)

    assert result.succeeded, result.message
    assert len(fake.read("Run_Log")) == 10


def test_pipeline_reports_failure_for_empty_source():
    fake = FakeSheetsService(worksheets={"Raw_Data": []}, spreadsheet_id="sheet-empty")
    settings = load_settings(env={**MOCK_AI, "GS_SPREADSHEET_ID": "sheet-empty"})
    result = run_pipeline(settings, sheets_service=fake)

    assert not result.succeeded
    assert result.workbook is None
    assert result.message


def test_build_trigger_matches_configuration():
    pytest.importorskip("apscheduler")
    common = {"day_of_week": 1, "interval_minutes": 30, "timezone": "UTC"}
    daily = build_trigger(
        SchedulerSettings(mode="daily", time_of_day=time(7, 30), **common)
    )
    weekly = build_trigger(
        SchedulerSettings(mode="weekly", time_of_day=time(8, 0), **common)
    )
    interval = build_trigger(
        SchedulerSettings(mode="interval", time_of_day=time(0, 0), **common)
    )
    assert daily is not None
    assert weekly is not None
    assert interval is not None


def test_cli_exit_codes(monkeypatch, tmp_path: Path):
    import main as cli

    settings = load_settings(env={**MOCK_AI, "LOCAL_OUTPUT_DIR": str(tmp_path)})
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: settings)
    assert cli.main(["--dry-run", "--demo"]) == 0
    assert (tmp_path / "Run_Log.csv").is_file()

    def _bad_settings(*args, **kwargs):
        raise ConfigurationError("bad config")

    monkeypatch.setattr(cli, "load_settings", _bad_settings)
    assert cli.main([]) == 2

    failed = PipelineResult(
        run_id="run-fail",
        outcome=OUTCOME_FAILED,
        message="boom",
        mode="dry_run",
        data_source="demo",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        duration_seconds=1.0,
    )
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: failed)
    assert cli.main([]) == 1