"""Pipeline orchestration: read -> quality -> analytics -> AI -> report -> sink."""

from __future__ import annotations

from src.pipeline.pipeline import (
    OUTCOME_FAILED,
    OUTCOME_SUCCESS,
    PipelineResult,
    run_pipeline,
)
from src.pipeline.scheduler import build_trigger, run_scheduler

__all__ = [
    "OUTCOME_FAILED",
    "OUTCOME_SUCCESS",
    "PipelineResult",
    "build_trigger",
    "run_pipeline",
    "run_scheduler",
]