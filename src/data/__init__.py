"""Data layer: synthetic demo data, Google Sheets <-> DataFrame IO, and the
deterministic validation / cleaning stages.

Nothing in this package talks to an external API - it operates purely on pandas
DataFrames, which is what makes it fully unit testable.
"""

from src.data.cleaning import (
    ORDER_MONTH_COLUMN,
    SOURCE_ROW_COLUMN,
    canonical_value,
    clean_frame,
    coerce_types,
    normalise_text,
    normalize_headers,
    parse_date,
    parse_integer,
    parse_number,
    prepare_frame,
)
from src.data.demo_data import DEMO_DATA_DISCLAIMER, DemoDataset, build_demo_dataset
from src.data.io import dataframe_from_values, is_blank, values_from_dataframe
from src.data.models import (
    CleaningAction,
    CleaningResult,
    CoercionFailures,
    RejectedRecord,
    ValidationIssue,
    ValidationResult,
)
from src.data.quality import QualityOutcome, run_quality_stages
from src.data.validation import ValidationPolicy, validate_frame

__all__ = [
    "DEMO_DATA_DISCLAIMER",
    "ORDER_MONTH_COLUMN",
    "SOURCE_ROW_COLUMN",
    "CleaningAction",
    "CleaningResult",
    "CoercionFailures",
    "DemoDataset",
    "QualityOutcome",
    "RejectedRecord",
    "ValidationIssue",
    "ValidationPolicy",
    "ValidationResult",
    "build_demo_dataset",
    "canonical_value",
    "clean_frame",
    "coerce_types",
    "dataframe_from_values",
    "is_blank",
    "normalise_text",
    "normalize_headers",
    "parse_date",
    "parse_integer",
    "parse_number",
    "prepare_frame",
    "run_quality_stages",
    "validate_frame",
    "values_from_dataframe",
]
