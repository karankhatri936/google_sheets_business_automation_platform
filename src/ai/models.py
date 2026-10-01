"""Typed results of the AI interpretation layer.

These dataclasses deliberately contain no provider code: the reporting layer can
label and display AI output without importing any AI client, and every AI artefact
carries an explicit flag/notice that it is AI-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

AI_GENERATED_NOTICE = "AI-generated content - review before acting on it."


class AIStatus(str, Enum):
    """Outcome of an AI feature within a run."""

    GENERATED = "generated"
    SKIPPED = "skipped"
    FAILED = "failed"

    def label(self) -> str:
        return self.value


@dataclass(frozen=True)
class AIInsights:
    """Natural-language interpretation of already-calculated metrics."""

    status: AIStatus
    status_detail: str
    provider_label: str = ""
    model: str = ""
    generated_at: str = ""
    executive_summary: str = ""
    positive_trends: tuple[str, ...] = ()
    negative_trends: tuple[str, ...] = ()
    watch_items: tuple[str, ...] = ()
    management_summary: str = ""
    unverified_numbers: tuple[float, ...] = ()
    metrics_supplied: tuple[str, ...] = ()
    is_ai_generated: bool = True

    @property
    def has_content(self) -> bool:
        return bool(
            self.executive_summary
            or self.positive_trends
            or self.negative_trends
            or self.watch_items
            or self.management_summary
        )

    @classmethod
    def disabled(cls, reason: str) -> AIInsights:
        """Placeholder used when the AI layer did not run (off/unconfigured)."""
        return cls(status=AIStatus.SKIPPED, status_detail=reason, is_ai_generated=False)


@dataclass(frozen=True)
class ClassificationRecord:
    """One classified free-text record."""

    source_row: int
    order_id: str
    text: str
    category: str

    @property
    def is_ai_generated(self) -> bool:
        return True


@dataclass(frozen=True)
class ClassificationSummary:
    """Outcome of the optional feedback classification feature."""

    status: AIStatus
    status_detail: str
    provider_label: str = ""
    model: str = ""
    categories: tuple[str, ...] = ()
    records: tuple[ClassificationRecord, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    unclassified: int = 0
    notices: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.records)

    def share_pct(self, category: str) -> float:
        if not self.records:
            return 0.0
        matched = sum(1 for record in self.records if record.category == category)
        return round(100.0 * matched / len(self.records), 1)
