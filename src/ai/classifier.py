"""Optional AI classification of the free-text feedback column.

Produces a :class:`ClassificationSummary` for the Feedback_Analysis worksheet.
The feature is guarded at every level (disabled, missing column, no texts,
provider failures) and always returns a summary - never raises - so the
pipeline can keep going and the worksheet explains why nothing was classified.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from src.ai.models import AIStatus, ClassificationRecord, ClassificationSummary
from src.ai.provider import (
    CATEGORIES_MARKER,
    ITEMS_MARKER,
    ChatProvider,
    build_provider,
    parse_json_response,
)
from src.config.logging_config import get_logger
from src.config.settings import AISettings
from src.data.io import is_blank
from src.utils.errors import AIConfigurationError, AIProviderError, AIResponseError

SYSTEM_PROMPT = (
    "You classify short customer feedback text into exactly one category from "
    "the allowed list supplied in the categories block below.\n"
    "Respond with ONLY a JSON array of objects of the form "
    '[{"index": <item index>, "category": "<one allowed category>"}] - no extra text.\n'
    "Never invent categories. If a text carries no signal, use the first allowed category."
)

_MAX_TEXT_CHARS = 500


def _skipped(settings: AISettings, detail: str) -> ClassificationSummary:
    return ClassificationSummary(
        status=AIStatus.SKIPPED,
        status_detail=detail,
        provider_label=settings.provider_label if settings.enabled else "",
        model=settings.model if settings.enabled else "",
        categories=tuple(settings.classification_categories),
    )


def _collect_candidates(
    frame: pd.DataFrame, column: str, max_rows: int
) -> list[tuple[int, str, str]]:
    """``(source_row, order_id, text)`` for the first ``max_rows`` non-blank texts."""
    candidates: list[tuple[int, str, str]] = []
    for _, row in frame.iterrows():
        text = row.get(column)
        if is_blank(text):
            continue
        try:
            source_row = int(row.get("source_row"))
        except (TypeError, ValueError):
            source_row = -1
        order_id = row.get("order_id")
        candidates.append(
            (source_row, "" if is_blank(order_id) else str(order_id), str(text))
        )
        if len(candidates) >= max_rows:
            break
    return candidates


def _counts(records: list[ClassificationRecord]) -> tuple[tuple[str, int], ...]:
    counter = Counter(record.category for record in records)
    return tuple(
        (category, count)
        for category, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def classify_feedback(
    settings: AISettings,
    clean_data: pd.DataFrame,
    *,
    provider: ChatProvider | None = None,
) -> ClassificationSummary:
    """Classify feedback texts into the configured categories (best effort)."""
    logger = get_logger("ai.classifier")
    if not settings.classification_enabled:
        return _skipped(
            settings, "classification disabled via AI_CLASSIFICATION_ENABLED=false"
        )
    reason = settings.unavailable_reason
    if reason is not None:
        return _skipped(settings, reason)

    column = settings.classification_text_column
    if column not in clean_data.columns:
        return _skipped(
            settings, f"text column {column!r} is not present in the cleaned dataset"
        )
    candidates = _collect_candidates(clean_data, column, settings.classification_max_rows)
    if not candidates:
        return _skipped(settings, f"no non-empty {column!r} values to classify")

    categories = tuple(settings.classification_categories)
    records: list[ClassificationRecord] = []
    unclassified = 0
    batch_size = settings.classification_batch_size
    try:
        active = provider if provider is not None else build_provider(settings)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            items = [
                {"index": position, "text": text[:_MAX_TEXT_CHARS]}
                for position, (_, _, text) in enumerate(batch)
            ]
            system = (
                f"{SYSTEM_PROMPT}\n{CATEGORIES_MARKER} "
                f"{json.dumps(list(categories))}"
            )
            user = f"{ITEMS_MARKER}\n{json.dumps(items)}"
            raw = active.complete(system=system, user=user)
            payload = parse_json_response(raw)
            if not isinstance(payload, list):
                raise AIResponseError("the classification response must be a JSON array")
            labels: dict[int, str] = {}
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                try:
                    labels[int(entry["index"])] = str(entry["category"])
                except (KeyError, TypeError, ValueError):
                    continue
            for position, (source_row, order_id, text) in enumerate(batch):
                label = labels.get(position)
                if label is None or label not in categories:
                    unclassified += 1
                    continue
                records.append(
                    ClassificationRecord(
                        source_row=source_row,
                        order_id=order_id,
                        text=text,
                        category=label,
                    )
                )
    except (AIConfigurationError, AIProviderError, AIResponseError) as exc:
        logger.warning("feedback classification failed: %s", exc)
        return ClassificationSummary(
            status=AIStatus.FAILED,
            status_detail=f"{type(exc).__name__}: {exc}",
            provider_label=settings.provider_label,
            model=settings.model,
            categories=categories,
            records=tuple(records),
            counts=_counts(records),
            unclassified=unclassified,
        )

    summary = ClassificationSummary(
        status=AIStatus.GENERATED,
        status_detail=(
            f"classified {len(records)} record(s) into "
            f"{len(_counts(records))} categor(ies)"
        ),
        provider_label=settings.provider_label,
        model=settings.model,
        categories=categories,
        records=tuple(records),
        counts=_counts(records),
        unclassified=unclassified,
    )
    logger.info(
        "feedback classification status=%s records=%d unclassified=%d",
        summary.status.value,
        len(records),
        unclassified,
    )
    return summary


__all__ = ["SYSTEM_PROMPT", "classify_feedback"]