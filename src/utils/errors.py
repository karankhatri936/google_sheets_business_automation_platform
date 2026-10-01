"""Explicit exception hierarchy for the platform.

Every failure mode the application knows about has a dedicated exception so the
pipeline can distinguish (and report) *why* something failed instead of guessing
from generic ``Exception`` messages.
"""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for all application specific errors."""


# --------------------------------------------------------------------------
# Configuration / authentication
# --------------------------------------------------------------------------
class ConfigurationError(PlatformError):
    """Raised when configuration is missing, malformed or contradictory."""


class AuthenticationError(PlatformError):
    """Raised when Google credentials cannot be loaded or are rejected."""


# --------------------------------------------------------------------------
# Google Sheets
# --------------------------------------------------------------------------
class GoogleSheetsError(PlatformError):
    """Base class for Google Sheets API failures."""


class SpreadsheetNotFoundError(GoogleSheetsError):
    """Raised when the configured spreadsheet cannot be accessed."""


class WorksheetNotFoundError(GoogleSheetsError):
    """Raised when a required worksheet (tab) does not exist."""


class SheetsApiError(GoogleSheetsError):
    """Raised when the Sheets API returns an error response."""


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------
class DataReadError(PlatformError):
    """Raised when raw data cannot be read or is structurally unusable."""


class ValidationError(PlatformError):
    """Raised when data violates the business schema in a fatal way.

    Non-fatal validation problems are reported through the validation report;
    this exception is reserved for conditions that make a run meaningless
    (for example an empty spreadsheet or all columns missing).
    """


class DataSinkError(PlatformError):
    """Raised when writing results to a sink (Sheets or local files) fails."""


# --------------------------------------------------------------------------
# AI interpretation layer
# --------------------------------------------------------------------------
class AIConfigurationError(ConfigurationError):
    """Raised when the AI layer is enabled but not correctly configured."""


class AIProviderError(PlatformError):
    """Raised when the configured AI provider call fails (network, HTTP, ...)."""


class AIResponseError(PlatformError):
    """Raised when an AI response cannot be parsed into the expected structure."""


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
class PipelineError(PlatformError):
    """Raised for unrecoverable pipeline orchestration failures."""
