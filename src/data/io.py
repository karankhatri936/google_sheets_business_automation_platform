"""Translation between Google Sheets value grids and pandas DataFrames.

The Google Sheets API works with ``list[list[Any]]`` value grids (the
"unformatted values" representation). This module owns the conversion in both
directions so no other layer needs to know about cell padding, header rows or
timestamp formatting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

# Index name used for every raw frame: the label is the spreadsheet row number.
SOURCE_ROW_INDEX = "source_row"

_EMPTY_TOKENS = frozenset({"", " ", "na", "n/a", "null", "none", "nan", "-"})


def is_blank(value: Any) -> bool:
    """True when a cell value should be treated as missing.

    Google Sheets returns empty cells as ``""`` inside a range request, so blank
    detection happens here and nowhere else.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _EMPTY_TOKENS:
        return True
    return False


def dataframe_from_values(
    values: Sequence[Sequence[Any]],
    *,
    first_sheet_row: int = 1,
) -> pd.DataFrame:
    """Convert a value grid into a raw DataFrame.

    Parameters
    ----------
    values:
        The grid as returned by the Sheets API (or loaded from a CSV export).
        Row 0 is expected to be the header row.
    first_sheet_row:
        Spreadsheet row number of ``values[0]``. Defaults to 1 (header row).

    Notes
    -----
    * Missing trailing cells are padded with ``None`` so every row has the same
      width (the Sheets API omits trailing empty cells).
    * Columns beyond the header width get explicit ``extra_column_N`` names
      instead of being dropped, so unexpected data is reported rather than lost.
    * The returned frame's index is the spreadsheet row number - it is the
      traceability key used by validation and the Data_Quality worksheet.
    """
    rows = [list(row) for row in values if row is not None]
    if not rows:
        return pd.DataFrame()

    header = [str(cell).strip() if not is_blank(cell) else "" for cell in rows[0]]
    data_rows = rows[1:]

    width = max([len(header)] + [len(row) for row in data_rows])
    header = header + [f"extra_column_{i}" for i in range(len(header) + 1, width + 1)]
    header = [
        name if name else f"unnamed_column_{position + 1}"
        for position, name in enumerate(header)
    ]

    padded = [row + [None] * (width - len(row)) for row in data_rows]
    index = list(range(first_sheet_row + 1, first_sheet_row + 1 + len(padded)))
    frame = pd.DataFrame(padded, columns=header, index=index)
    frame.index.name = SOURCE_ROW_INDEX
    return frame


def _cell_value(value: Any) -> Any:
    """Convert a single pandas cell into a Sheets-API friendly value."""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return int(value) if value.is_integer() else round(value, 2)
    if isinstance(value, str):
        return value
    return str(value)


def values_from_dataframe(frame: pd.DataFrame, *, include_index: bool = False) -> list[list[Any]]:
    """Convert a DataFrame into a value grid: header row first, then data."""
    rows: list[list[Any]] = [[str(column) for column in frame.columns]]
    for _, record in frame.iterrows():
        row = [_cell_value(value) for value in record.tolist()]
        if include_index:
            row.insert(0, _cell_value(record.name))
        rows.append(row)
    return rows


def frame_from_records(records: Sequence[dict[str, Any]], *, start_row: int = 2) -> pd.DataFrame:
    """Build a raw frame from records (used by the synthetic demo dataset)."""
    frame = pd.DataFrame(list(records))
    frame.index = range(start_row, start_row + len(frame))
    frame.index.name = SOURCE_ROW_INDEX
    return frame
