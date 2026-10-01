"""Typed wrapper around the Google Sheets API v4 service.

The client is deliberately thin: it owns request construction and error
translation, nothing else. Business decisions live in the data/analytics layers.

It receives the raw ``googleapiclient`` service object through its constructor,
which keeps the class fully testable with a fake service (see
``tests/fixtures/fake_sheets_service.py``) - no test needs credentials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from googleapiclient.errors import HttpError

from src.config.logging_config import get_logger
from src.utils.errors import (
    AuthenticationError,
    ConfigurationError,
    GoogleSheetsError,
    SheetsApiError,
    SpreadsheetNotFoundError,
    WorksheetNotFoundError,
)

USER_ENTERED = "USER_ENTERED"
RAW = "RAW"
CellValue = Any
ValueGrid = Sequence[Sequence[CellValue]]


class GoogleSheetsClient:
    """Read/write access to one spreadsheet."""

    def __init__(self, service: Any, spreadsheet_id: str) -> None:
        if not spreadsheet_id or not spreadsheet_id.strip():
            raise ConfigurationError("a spreadsheet id is required to access Google Sheets")
        self._service = service
        self._spreadsheet_id = spreadsheet_id.strip()
        self._worksheet_ids: dict[str, int] | None = None
        self._logger = get_logger("google.sheets")

    @property
    def spreadsheet_id(self) -> str:
        return self._spreadsheet_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_values(self, a1_range: str) -> list[list[CellValue]]:
        """Read a range in unformatted-value mode (empty trailing cells omitted)."""
        request = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=self._spreadsheet_id, range=a1_range, majorDimension="ROWS")
        )
        response = self._execute(request, f"reading range {a1_range}")
        values = response.get("values", []) if isinstance(response, dict) else []
        self._logger.info("read %d row(s) from range %s", len(values), a1_range)
        return [list(row) for row in values]

    def get_spreadsheet_metadata(self) -> dict[str, Any]:
        """Spreadsheet properties including every worksheet."""
        request = self._service.spreadsheets().get(
            spreadsheetId=self._spreadsheet_id, includeGridData=False
        )
        response = self._execute(request, "reading spreadsheet metadata")
        return response if isinstance(response, dict) else {}

    def list_worksheets(self) -> tuple[str, ...]:
        """Titles of all worksheets, in spreadsheet order."""
        metadata = self.get_spreadsheet_metadata()
        return tuple(
            str(sheet["properties"]["title"])
            for sheet in metadata.get("sheets", [])
            if "properties" in sheet
        )

    def worksheet_ids(self, *, refresh: bool = False) -> dict[str, int]:
        """Mapping of worksheet title -> numeric sheet id (cached per instance)."""
        if self._worksheet_ids is None or refresh:
            metadata = self.get_spreadsheet_metadata()
            self._worksheet_ids = {
                str(sheet["properties"]["title"]): int(sheet["properties"]["sheetId"])
                for sheet in metadata.get("sheets", [])
                if "properties" in sheet
            }
        return dict(self._worksheet_ids)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write_values(
        self,
        a1_range: str,
        values: ValueGrid,
        *,
        value_input_option: str = USER_ENTERED,
    ) -> int:
        """Overwrite a range and return the number of updated cells."""
        body = {"range": a1_range, "majorDimension": "ROWS", "values": [list(r) for r in values]}
        request = (
            self._service.spreadsheets()
            .values()
            .update(
                spreadsheetId=self._spreadsheet_id,
                range=a1_range,
                valueInputOption=value_input_option,
                body=body,
            )
        )
        response = self._execute(request, f"writing range {a1_range}")
        return int(response.get("updatedCells", 0)) if isinstance(response, dict) else 0

    def append_values(
        self,
        a1_range: str,
        values: ValueGrid,
        *,
        value_input_option: str = USER_ENTERED,
    ) -> int:
        """Append rows after the last table row and return the updated cell count."""
        body = {"majorDimension": "ROWS", "values": [list(r) for r in values]}
        request = (
            self._service.spreadsheets()
            .values()
            .append(
                spreadsheetId=self._spreadsheet_id,
                range=a1_range,
                valueInputOption=value_input_option,
                insertDataOption="INSERT_ROWS",
                body=body,
            )
        )
        response = self._execute(request, f"appending to range {a1_range}")
        if isinstance(response, dict):
            return int(response.get("updates", {}).get("updatedCells", 0))
        return 0

    def clear_values(self, a1_range: str) -> None:
        """Clear the values of a range (formatting is preserved)."""
        request = (
            self._service.spreadsheets()
            .values()
            .clear(spreadsheetId=self._spreadsheet_id, range=a1_range, body={})
        )
        self._execute(request, f"clearing range {a1_range}")

    def batch_write_values(
        self,
        ranges: Mapping[str, ValueGrid],
        *,
        value_input_option: str = USER_ENTERED,
    ) -> int:
        """Write several ranges in one API call; returns the total updated cells."""
        if not ranges:
            return 0
        body = {
            "valueInputOption": value_input_option,
            "data": [
                {
                    "range": a1_range,
                    "majorDimension": "ROWS",
                    "values": [list(row) for row in values],
                }
                for a1_range, values in ranges.items()
            ],
        }
        request = self._service.spreadsheets().values().batchUpdate(
            spreadsheetId=self._spreadsheet_id, body=body
        )
        response = self._execute(request, "batch writing ranges")
        total = int(response.get("totalUpdatedCells", 0)) if isinstance(response, dict) else 0
        self._logger.info("batch wrote %d range(s), %d cell(s)", len(ranges), total)
        return total

    # ------------------------------------------------------------------
    # Worksheet management / formatting
    # ------------------------------------------------------------------
    def ensure_worksheet(self, title: str, *, rows: int = 1000, columns: int = 26) -> int:
        """Create the worksheet when missing and return its numeric sheet id."""
        existing = self.worksheet_ids()
        if title in existing:
            return existing[title]

        body = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": title,
                            "gridProperties": {"rowCount": rows, "columnCount": columns},
                        }
                    }
                }
            ]
        }
        request = self._service.spreadsheets().batchUpdate(
            spreadsheetId=self._spreadsheet_id, body=body
        )
        try:
            response = self._execute(request, f"creating worksheet {title}")
        except GoogleSheetsError:
            # A concurrent run may have created it - re-read before failing.
            refreshed = self.worksheet_ids(refresh=True)
            if title in refreshed:
                return refreshed[title]
            raise

        sheet_id = self._extract_created_sheet_id(response, title)
        self._worksheet_ids = {**self.worksheet_ids(), title: sheet_id}
        self._logger.info("created worksheet %s (sheetId=%s)", title, sheet_id)
        return sheet_id

    def apply_batch_update(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply a list of ``batchUpdate`` requests (used by the formatter)."""
        if not requests:
            return {}
        request = self._service.spreadsheets().batchUpdate(
            spreadsheetId=self._spreadsheet_id, body={"requests": [dict(r) for r in requests]}
        )
        response = self._execute(request, "applying spreadsheet formatting")
        return response if isinstance(response, dict) else {}

    def _extract_created_sheet_id(self, response: Any, title: str) -> int:
        if isinstance(response, dict):
            replies = response.get("replies", [])
            if replies and isinstance(replies[0], dict):
                properties = replies[0].get("addSheet", {}).get("properties", {})
                if "sheetId" in properties:
                    return int(properties["sheetId"])
        raise SheetsApiError(f"could not determine the sheet id of the new worksheet {title!r}")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _execute(self, request: Any, context: str) -> Any:
        try:
            return request.execute()
        except HttpError as exc:
            raise self._translate(exc, context) from exc

    def _translate(self, error: HttpError, context: str) -> GoogleSheetsError:
        status = getattr(error, "status_code", None)
        message = getattr(error, "reason", None) or str(error)
        lowered = message.lower()

        if status == 404:
            return SpreadsheetNotFoundError(
                "spreadsheet not found or not shared with the configured credentials "
                f"(id starts with {self._spreadsheet_id[:6]}...)"
            )
        if status == 400 and "unable to parse range" in lowered:
            return WorksheetNotFoundError(f"worksheet/range not found while {context}: {message}")
        if status in (401, 403):
            return AuthenticationError(
                f"Google Sheets denied access while {context} (HTTP {status}): {message}"
            )
        return SheetsApiError(f"Google Sheets API error while {context} (HTTP {status}): {message}")
