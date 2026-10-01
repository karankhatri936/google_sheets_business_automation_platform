"""Formatting facade."""

from __future__ import annotations

from typing import Any

from src.google_sheets.formatting import (
    build_formatting_requests,
    build_workbook_formatting_requests,
)
from src.reporting.models import SheetTable


class SheetFormatter:
    """Facade for turning SheetTable specifications into Google Sheets requests."""

    @staticmethod
    def format_table(table: SheetTable, sheet_id: int) -> list[dict[str, Any]]:
        return build_formatting_requests(table, sheet_id)

    @staticmethod
    def format_workbook(
        tables: tuple[SheetTable, ...], sheet_ids: dict[str, int]
    ) -> list[dict[str, Any]]:
        return build_workbook_formatting_requests(tables, sheet_ids)


__all__ = ["SheetFormatter"]
