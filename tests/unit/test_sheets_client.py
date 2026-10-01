"""Tests for the Google Sheets API client and authentication configuration.

Everything here runs against the in-memory fake service - no credentials, no
network access. The real API is never contacted by the unit test suite.
"""

from __future__ import annotations

import pytest

from src.google_sheets.auth import build_credentials
from src.google_sheets.client import GoogleSheetsClient
from src.utils.errors import (
    AuthenticationError,
    ConfigurationError,
    SheetsApiError,
    SpreadsheetNotFoundError,
    WorksheetNotFoundError,
)
from tests.fixtures.fake_sheets_service import FakeSheetsService

SPREADSHEET_ID = "fake-spreadsheet-id"

HEADER = ["date", "order_id", "revenue"]
TABLE = [
    HEADER,
    ["2025-01-01", "SO-1", 100.0],
    ["2025-01-02", "SO-2", 250.5],
]


@pytest.fixture
def service() -> FakeSheetsService:
    return FakeSheetsService({"Raw_Data": TABLE, "Run_Log": []})


@pytest.fixture
def client(service: FakeSheetsService) -> GoogleSheetsClient:
    return GoogleSheetsClient(service, SPREADSHEET_ID)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_read_values_returns_rows_and_uses_rows_dimension(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    values = client.read_values("Raw_Data!A1:Z100")
    assert values == TABLE
    params = service.calls_for("values.get")[0]
    assert params["spreadsheetId"] == SPREADSHEET_ID
    assert params["range"] == "Raw_Data!A1:Z100"
    assert params["majorDimension"] == "ROWS"


def test_list_worksheets_and_ids(client: GoogleSheetsClient) -> None:
    assert client.list_worksheets() == ("Raw_Data", "Run_Log")
    assert client.worksheet_ids() == {"Raw_Data": 0, "Run_Log": 1}


def test_worksheet_ids_are_cached(client: GoogleSheetsClient, service: FakeSheetsService) -> None:
    client.worksheet_ids()
    client.worksheet_ids()
    assert len(service.calls_for("spreadsheets.get")) == 1
    client.worksheet_ids(refresh=True)
    assert len(service.calls_for("spreadsheets.get")) == 2


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_write_values_overwrites_range(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    updated = client.write_values("Raw_Data!A1", TABLE)
    assert updated == 9
    assert service.read("Raw_Data") == TABLE
    body = service.calls_for("values.update")[0]["body"]
    assert body["range"] == "Raw_Data!A1"
    assert body["majorDimension"] == "ROWS"


def test_append_values_appends_below_existing_rows(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    updated = client.append_values("Run_Log!A1", [["run-1", "SUCCESS"]])
    assert updated == 2
    assert service.read("Run_Log") == [["run-1", "SUCCESS"]]
    params = service.calls_for("values.append")[0]
    assert params["insertDataOption"] == "INSERT_ROWS"


def test_batch_write_values_sends_one_request(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    total = client.batch_write_values({"Run_Log!A1": [["run-1"]], "Raw_Data!A1": [["a", "b"]]})
    assert total == 3
    assert len(service.calls_for("values.batchUpdate")) == 1
    body = service.calls_for("values.batchUpdate")[0]["body"]
    assert [entry["range"] for entry in body["data"]] == ["Run_Log!A1", "Raw_Data!A1"]
    assert body["valueInputOption"] == "USER_ENTERED"


def test_clear_values(client: GoogleSheetsClient, service: FakeSheetsService) -> None:
    client.clear_values("Raw_Data!A:Z")
    assert service.read("Raw_Data") == []
    assert service.calls_for("values.clear")[0]["range"] == "Raw_Data!A:Z"


# ---------------------------------------------------------------------------
# Worksheet management
# ---------------------------------------------------------------------------
def test_ensure_worksheet_returns_existing_id(client: GoogleSheetsClient) -> None:
    assert client.ensure_worksheet("Raw_Data") == 0


def test_ensure_worksheet_creates_missing_sheet(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    sheet_id = client.ensure_worksheet("KPI_Summary", rows=500, columns=12)
    assert sheet_id == 2
    assert "KPI_Summary" in service.worksheets
    request_body = service.calls_for("spreadsheets.batchUpdate")[0]["body"]
    properties = request_body["requests"][0]["addSheet"]["properties"]
    assert properties["title"] == "KPI_Summary"
    assert properties["gridProperties"] == {"rowCount": 500, "columnCount": 12}
    # The cache is refreshed so a second call does not create it again.
    assert client.ensure_worksheet("KPI_Summary") == sheet_id


def test_apply_batch_update_sends_requests(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    client.apply_batch_update([{"repeatCell": {"range": {}}}])
    body = service.calls_for("spreadsheets.batchUpdate")[0]["body"]
    assert body["requests"] == [{"repeatCell": {"range": {}}}]


def test_apply_batch_update_with_no_requests_is_a_noop(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    assert client.apply_batch_update([]) == {}
    assert service.calls_for("spreadsheets.batchUpdate") == []


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------
def test_spreadsheet_not_found_is_translated(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    service.fail_next("spreadsheets.get", 404, "Requested entity was not found.")
    with pytest.raises(SpreadsheetNotFoundError):
        client.get_spreadsheet_metadata()


def test_invalid_range_is_reported_as_missing_worksheet(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    service.fail_next("values.get", 400, "Unable to parse range: Missing!A1:Z10")
    with pytest.raises(WorksheetNotFoundError):
        client.read_values("Missing!A1:Z10")


def test_permission_error_is_reported_as_authentication_failure(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    service.fail_next("values.get", 403, "The caller does not have permission")
    with pytest.raises(AuthenticationError):
        client.read_values("Raw_Data!A1:Z10")


def test_other_api_errors_are_wrapped(
    client: GoogleSheetsClient, service: FakeSheetsService
) -> None:
    service.fail_next("values.update", 500, "Backend Error")
    with pytest.raises(SheetsApiError):
        client.write_values("Raw_Data!A1", TABLE)


def test_client_requires_a_spreadsheet_id(service: FakeSheetsService) -> None:
    with pytest.raises(ConfigurationError):
        GoogleSheetsClient(service, "   ")


# ---------------------------------------------------------------------------
# Authentication configuration (no live credentials involved)
# ---------------------------------------------------------------------------
def test_missing_service_account_file_is_reported_clearly(tmp_path) -> None:
    from src.config.settings import GoogleAuthSettings

    auth = GoogleAuthSettings(
        mode="service_account",
        service_account_file=tmp_path / "does-not-exist.json",
        service_account_json=None,
        oauth_client_secrets_file=tmp_path / "client_secret.json",
        oauth_token_file=tmp_path / "token.json",
        scopes=("https://www.googleapis.com/auth/spreadsheets",),
    )
    with pytest.raises(AuthenticationError, match="service-account key file not found"):
        build_credentials(auth)


def test_invalid_service_account_json_is_reported_clearly(tmp_path) -> None:
    from src.config.settings import GoogleAuthSettings

    auth = GoogleAuthSettings(
        mode="service_account",
        service_account_file=tmp_path / "unused.json",
        service_account_json="{not json",
        oauth_client_secrets_file=tmp_path / "client_secret.json",
        oauth_token_file=tmp_path / "token.json",
        scopes=("https://www.googleapis.com/auth/spreadsheets",),
    )
    with pytest.raises(AuthenticationError, match="valid JSON"):
        build_credentials(auth)


def test_missing_oauth_client_secrets_is_reported_clearly(tmp_path) -> None:
    from src.config.settings import GoogleAuthSettings

    auth = GoogleAuthSettings(
        mode="oauth",
        service_account_file=tmp_path / "unused.json",
        service_account_json=None,
        oauth_client_secrets_file=tmp_path / "client_secret.json",
        oauth_token_file=tmp_path / "token.json",
        scopes=("https://www.googleapis.com/auth/spreadsheets",),
    )
    with pytest.raises(AuthenticationError, match="OAuth client secrets file not found"):
        build_credentials(auth)
