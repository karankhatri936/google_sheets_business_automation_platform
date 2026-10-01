"""In-memory stand-in for the ``googleapiclient`` Sheets v4 service.

The fake mirrors the *shape* of the real discovery client
(``service.spreadsheets().values().get(...)``), so the production client code
under test is exercised exactly as it runs against Google - only the transport is
replaced. It also records every call and can inject HTTP failures.

No network access, no credentials, no live spreadsheet.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from googleapiclient.errors import HttpError


class FakeHttpResponse:
    """The small part of ``httplib2.Response`` that ``HttpError`` reads."""

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason


def make_http_error(status: int, message: str, reason: str = "error") -> HttpError:
    """Build a real :class:`HttpError` so error translation is tested for real."""
    body = json.dumps({"error": {"code": status, "message": message, "status": reason}})
    return HttpError(FakeHttpResponse(status, reason), body.encode("utf-8"), uri="fake://sheets")


@dataclass
class RecordedCall:
    """One recorded API call."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)


class _FakeRequest:
    """Request object exposing the single ``execute()`` method the client uses."""

    def __init__(self, service: "FakeSheetsService", method: str) -> None:
        self._service = service
        self._method = method

    def execute(self) -> Any:
        error = self._service.failures.pop(self._method, None)
        if error is not None:
            raise error
        return copy.deepcopy(self._service.responses.get(self._method, {}))


def _sheet_title(a1_range: str) -> str:
    return a1_range.split("!", 1)[0].strip("'") if "!" in a1_range else a1_range


class _FakeValuesResource:
    def __init__(self, service: "FakeSheetsService") -> None:
        self._service = service

    def get(self, **params: Any) -> _FakeRequest:
        self._service.record("values.get", params)
        title = _sheet_title(str(params.get("range", "")))
        self._service.responses["values.get"] = {"values": self._service.read(title)}
        return self._service.request("values.get")

    def update(self, **params: Any) -> _FakeRequest:
        self._service.record("values.update", params)
        values = [list(row) for row in params.get("body", {}).get("values", [])]
        self._service.write(_sheet_title(str(params.get("range", ""))), values)
        self._service.responses["values.update"] = {
            "updatedCells": sum(len(row) for row in values),
            "updatedRange": str(params.get("range", "")),
        }
        return self._service.request("values.update")

    def append(self, **params: Any) -> _FakeRequest:
        self._service.record("values.append", params)
        values = [list(row) for row in params.get("body", {}).get("values", [])]
        title = _sheet_title(str(params.get("range", "")))
        self._service.write(title, self._service.read(title) + values)
        self._service.responses["values.append"] = {
            "updates": {"updatedCells": sum(len(row) for row in values)}
        }
        return self._service.request("values.append")

    def clear(self, **params: Any) -> _FakeRequest:
        self._service.record("values.clear", params)
        self._service.write(_sheet_title(str(params.get("range", ""))), [])
        self._service.responses["values.clear"] = {"clearedRange": str(params.get("range", ""))}
        return self._service.request("values.clear")

    def batchUpdate(self, **params: Any) -> _FakeRequest:
        self._service.record("values.batchUpdate", params)
        data = params.get("body", {}).get("data", [])
        for entry in data:
            self._service.write(
                _sheet_title(str(entry.get("range", ""))),
                [list(row) for row in entry.get("values", [])],
            )
        self._service.responses["values.batchUpdate"] = {
            "totalUpdatedCells": sum(len(row) for entry in data for row in entry.get("values", []))
        }
        return self._service.request("values.batchUpdate")


class _FakeSpreadsheetsResource:
    def __init__(self, service: "FakeSheetsService") -> None:
        self._service = service
        self._values = _FakeValuesResource(service)

    def values(self) -> _FakeValuesResource:
        return self._values

    def get(self, **params: Any) -> _FakeRequest:
        self._service.record("spreadsheets.get", params)
        self._service.responses["spreadsheets.get"] = self._service.metadata()
        return self._service.request("spreadsheets.get")

    def batchUpdate(self, **params: Any) -> _FakeRequest:
        self._service.record("spreadsheets.batchUpdate", params)
        replies: list[dict[str, Any]] = []
        for request in params.get("body", {}).get("requests", []):
            if "addSheet" in request:
                title = str(request["addSheet"]["properties"]["title"])
                sheet_id = self._service.add_sheet(title, request["addSheet"]["properties"])
                replies.append({"addSheet": {"properties": {"sheetId": sheet_id, "title": title}}})
            else:
                replies.append({})
        self._service.responses["spreadsheets.batchUpdate"] = {"replies": replies}
        return self._service.request("spreadsheets.batchUpdate")


class FakeSheetsService:
    """Records calls, serves in-memory worksheet data and injects failures."""

    def __init__(
        self,
        worksheets: dict[str, list[list[Any]]] | None = None,
        spreadsheet_id: str = "fake-spreadsheet-id",
    ) -> None:
        self.spreadsheet_id = spreadsheet_id
        self.worksheets: dict[str, list[list[Any]]] = (
            {title: [list(row) for row in rows] for title, rows in worksheets.items()}
            if worksheets
            else {"Sheet1": []}
        )
        self.sheet_ids: dict[str, int] = {
            title: index for index, title in enumerate(self.worksheets)
        }
        self.calls: list[RecordedCall] = []
        self.failures: dict[str, HttpError] = {}
        self.responses: dict[str, Any] = {}
        self._next_sheet_id = len(self.sheet_ids)

    # -- test helpers ---------------------------------------------------
    def record(self, method: str, params: dict[str, Any]) -> None:
        self.calls.append(RecordedCall(method, params))

    def request(self, method: str) -> _FakeRequest:
        return _FakeRequest(self, method)

    def fail_next(self, method: str, status: int = 500, message: str = "boom") -> None:
        """Make the next call to ``method`` fail with a real ``HttpError``."""
        self.failures[method] = make_http_error(status, message)

    def calls_for(self, method: str) -> list[dict[str, Any]]:
        return [call.params for call in self.calls if call.method == method]

    def read(self, title: str) -> list[list[Any]]:
        return self.worksheets.get(title, [])

    def write(self, title: str, values: list[list[Any]]) -> None:
        self.worksheets[title] = values

    def add_sheet(self, title: str, properties: dict[str, Any]) -> int:
        sheet_id = self._next_sheet_id
        self._next_sheet_id += 1
        self.worksheets.setdefault(title, [])
        self.sheet_ids[title] = sheet_id
        return sheet_id

    def metadata(self) -> dict[str, Any]:
        return {
            "spreadsheetId": self.spreadsheet_id,
            "properties": {"title": "Fake Spreadsheet"},
            "sheets": [
                {
                    "properties": {
                        "sheetId": self.sheet_ids[title],
                        "title": title,
                        "gridProperties": {"rowCount": 1000, "columnCount": 26},
                    }
                }
                for title in self.worksheets
            ],
        }

    # -- service surface -------------------------------------------------
    def spreadsheets(self) -> _FakeSpreadsheetsResource:
        return _FakeSpreadsheetsResource(self)
