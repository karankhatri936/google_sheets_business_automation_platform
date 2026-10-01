"""Shared pytest fixtures.

The default test suite is fully offline and deterministic: it uses the synthetic
demo dataset and never touches Google APIs or an AI provider.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.demo_data import DemoDataset, build_demo_dataset
from src.data.io import dataframe_from_values
from src.data.quality import QualityOutcome, run_quality_stages
from src.data.validation import ValidationPolicy

DEMO_OUTLIER_THRESHOLD = 10_000.0


@pytest.fixture(scope="session")
def demo_dataset() -> DemoDataset:
    """The synthetic dataset (cached for the whole test session)."""
    return build_demo_dataset()


@pytest.fixture
def demo_values(demo_dataset: DemoDataset) -> list[list[object]]:
    return demo_dataset.to_values()


@pytest.fixture
def demo_raw_frame(demo_values: list[list[object]]) -> pd.DataFrame:
    """Raw frame exactly as it would arrive from the Google Sheets API."""
    return dataframe_from_values(demo_values)


@pytest.fixture
def validation_policy() -> ValidationPolicy:
    return ValidationPolicy(revenue_outlier_threshold=DEMO_OUTLIER_THRESHOLD)


@pytest.fixture
def quality(
    demo_raw_frame: pd.DataFrame, validation_policy: ValidationPolicy
) -> QualityOutcome:
    """Prepared / validated / cleaned demo data."""
    return run_quality_stages(demo_raw_frame, policy=validation_policy)


@pytest.fixture
def clean_data(quality: QualityOutcome) -> pd.DataFrame:
    return quality.clean_data
