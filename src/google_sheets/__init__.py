"""Google Sheets integration: authentication, API client and formatting.

This package is the only place in the application that knows about the Google
Sheets API. Everything else works with plain pandas DataFrames and value grids.
"""

from src.google_sheets.auth import build_credentials, build_sheets_service
from src.google_sheets.client import GoogleSheetsClient

__all__ = [
    "GoogleSheetsClient",
    "build_credentials",
    "build_sheets_service",
]
