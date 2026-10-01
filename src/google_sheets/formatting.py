"""Translate report formatting hints into Google Sheets ``batchUpdate`` requests.

Formatting is intentionally restrained - it should make the workbook easy to read
for a non-technical business user, not decorate it:

* the header rows are bold with a light background,
* rows above the header can be frozen,
* money / integer / percentage cells get number formats,
* the requested columns get an explicit pixel width and text wrapping.

The output is a plain list of request dictionaries, which makes it fully testable
without touching the API.
"""

from __future__ import annotations

from typing import Any

from src.reporting.models import HEADER_BACKGROUND, SheetTable


def _column_range(sheet_id: int, table: SheetTable) -> dict[str, Any]:
    return {
        "sheetId": sheet_id,
        "startRowIndex": 0,
        "endRowIndex": max(table.row_count, 1),
        "startColumnIndex": 0,
        "endColumnIndex": max(table.column_count, 1),
    }


def _header_requests(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for row_index in table.header_rows:
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_index,
                        "endRowIndex": row_index + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": max(table.column_count, 1),
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True},
                            "backgroundColor": HEADER_BACKGROUND,
                        }
                    },
                    "fields": (
                        "userEnteredFormat.textFormat.bold,userEnteredFormat.backgroundColor"
                    ),
                }
            }
        )
    return requests


def _freeze_request(table: SheetTable, sheet_id: int) -> dict[str, Any] | None:
    if table.freeze_rows <= 0:
        return None
    return {
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {"frozenRowCount": table.freeze_rows},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    }


def _number_format_requests(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
    return [
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": cell.row,
                    "endRowIndex": cell.row + 1,
                    "startColumnIndex": cell.column,
                    "endColumnIndex": cell.column + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "numberFormat": {"type": cell.number_type, "pattern": cell.pattern}
                    }
                },
                "fields": "userEnteredFormat.numberFormat",
            }
        }
        for cell in table.number_formats
    ]


def _width_requests(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
    return [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": column,
                    "endIndex": column + 1,
                },
                "properties": {"pixelSize": pixels},
                "fields": "pixelSize",
            }
        }
        for column, pixels in table.column_widths
    ]


def _wrap_requests(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
    return [
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": max(table.freeze_rows, 0),
                    "endRowIndex": max(table.row_count, 1),
                    "startColumnIndex": column,
                    "endColumnIndex": column + 1,
                },
                "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                "fields": "userEnteredFormat.wrapStrategy",
            }
        }
        for column in table.wrap_columns
    ]


def build_formatting_requests(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
    """All formatting requests for one worksheet (stable order)."""
    requests: list[dict[str, Any]] = []
    freeze = _freeze_request(table, sheet_id)
    if freeze is not None:
        requests.append(freeze)
    requests.extend(_header_requests(table, sheet_id))
    requests.extend(_number_format_requests(table, sheet_id))
    requests.extend(_width_requests(table, sheet_id))
    requests.extend(_wrap_requests(table, sheet_id))
    return requests


def build_workbook_formatting_requests(
    tables: tuple[SheetTable, ...],
    sheet_ids: dict[str, int],
) -> list[dict[str, Any]]:
    """Formatting requests for every table that has a known sheet id."""
    requests: list[dict[str, Any]] = []
    for table in tables:
        sheet_id = sheet_ids.get(table.name)
        if sheet_id is None:
            continue
        requests.extend(build_formatting_requests(table, sheet_id))
    return requests


__all__ = [
    "build_formatting_requests",
    "build_workbook_formatting_requests",
]
