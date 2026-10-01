"""Data quality pipeline: the ordering of the three deterministic stages.

The sequence is deliberately explicit and lives in one place so the
orchestrator, the CLI and the tests all exercise the same rules::

    raw frame
      -> prepare_frame   (blank rows, headers, whitespace)
      -> coerce_types    (dates, integers, numbers)
      -> validate_frame  (invalid / repairable / valid)
      -> clean_frame     (aliases, defaults, derived fields, duplicates)
      -> business ready frame
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from src.config.schema import BUSINESS_SCHEMA, ColumnSpec
from src.config.logging_config import get_logger
from src.data.cleaning import clean_frame, coerce_types, prepare_frame
from src.data.models import CleaningAction, CleaningResult, CoercionFailures, ValidationResult
from src.data.validation import ValidationPolicy, validate_frame


@dataclass(frozen=True)
class QualityOutcome:
    """Everything the data quality stages produced for one raw dataset."""

    prepared: pd.DataFrame
    preparation_actions: tuple[CleaningAction, ...]
    coercion_failures: CoercionFailures
    validation: ValidationResult
    cleaning: CleaningResult

    @property
    def clean_data(self) -> pd.DataFrame:
        return self.cleaning.data


def run_quality_stages(
    raw_frame: pd.DataFrame,
    *,
    policy: ValidationPolicy | None = None,
    schema: Iterable[ColumnSpec] = BUSINESS_SCHEMA,
) -> QualityOutcome:
    """Run preparation, coercion, validation and cleaning in the documented order.

    Raises
    ------
    ValidationError
        Propagated from :func:`validate_frame` for fatal data problems.
    """
    logger = get_logger("data")
    prepared, preparation_actions = prepare_frame(raw_frame, schema)
    coerced, failures = coerce_types(prepared, schema)
    validation = validate_frame(coerced, failures, policy, schema)
    cleaning = clean_frame(coerced, validation, schema)
    logger.info(
        "data quality stages complete: %d prepared row(s), %d rejected, %d cleaned row(s)",
        len(prepared),
        validation.rejected_rows,
        cleaning.rows_out,
    )
    return QualityOutcome(
        prepared=prepared,
        preparation_actions=preparation_actions,
        coercion_failures=failures,
        validation=validation,
        cleaning=cleaning,
    )
