"""Deterministic facts -> natural-language interpretation (``AIInsights``).

The interpreter sends the KPI engine's output (a compact JSON digest) to the
configured provider and parses the narrative back into :class:`AIInsights`.
It never raises: provider/response failures are reported as a ``FAILED``
insights object so the AI_Insights worksheet is always written.

Numbers in the narrative that cannot be matched to a supplied metric are
surfaced through ``unverified_numbers`` (the workbook's Verification section).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from src.ai.models import AIInsights, AIStatus
from src.ai.provider import (
    FACTS_MARKER,
    ChatProvider,
    build_provider,
    parse_json_response,
)
from src.analytics.models import AnalyticsResult
from src.config.logging_config import get_logger
from src.config.settings import AISettings
from src.utils.errors import AIConfigurationError, AIProviderError, AIResponseError

SYSTEM_PROMPT = (
    "You are a business analyst. You receive verified metrics computed by a "
    "deterministic Python engine (never by a model) inside a FACTS_JSON block. "
    "Write a concise interpretation for a non-technical manager.\n"
    "Rules:\n"
    "* Only use numbers that appear in FACTS_JSON - never invent or recompute a figure.\n"
    "* Every string must be a short, self-contained sentence.\n"
    "* Respond with ONLY a JSON object of the form "
    '{"executive_summary": str, "positive_trends": [str], "negative_trends": [str], '
    '"watch_items": [str], "management_summary": str} with no extra text.'
)

# Optional sections shed (in this order) when the digest must fit max_input_chars.
_SHRINK_ORDER = ("period_change", "monthly_trend", "top_products", "status_distribution")

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
_MAX_LIST_ITEMS = 8
_MAX_STRING_CHARS = 300
_MAX_UNVERIFIED = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_facts_digest(
    analytics: AnalyticsResult, max_chars: int
) -> tuple[str, dict[str, Any]]:
    """Compact, JSON-serialisable digest of the KPI output.

    Optional sections are shed one entry at a time until the digest fits
    ``max_chars``; the scalar metrics are always included so the model always
    receives the headline numbers.
    """
    overall = analytics.overall
    facts: dict[str, Any] = {
        "period": analytics.period_label,
        "net_revenue": round(overall.net_revenue, 2),
        "gross_revenue": round(overall.gross_revenue, 2),
        "order_count": overall.order_count,
        "revenue_orders": overall.revenue_orders,
        "distinct_customers": overall.distinct_customers,
        "total_quantity": overall.total_quantity,
        "average_order_value": round(overall.average_order_value, 2),
        "average_unit_price": round(overall.average_unit_price, 2),
        "cancellation_rate_pct": overall.cancellation_rate_pct,
        "top_products": [
            {"product": m.key, "net_revenue": round(m.net_revenue, 2)}
            for m in analytics.by_product
        ],
        "monthly_trend": [
            {"period": p.label, "orders": p.order_count, "net_revenue": round(p.net_revenue, 2)}
            for p in analytics.trend
        ],
        "period_change": [
            {
                "metric": c.metric,
                "current": round(c.current, 2),
                "previous": round(c.previous, 2),
                "change": round(c.change, 2),
                "change_pct": None if c.change_pct is None else round(c.change_pct, 1),
            }
            for c in analytics.comparisons
        ],
        "status_distribution": [
            {"status": status, "orders": count}
            for status, count in analytics.status_distribution
        ],
    }

    def _over_budget() -> bool:
        return len(json.dumps(facts, sort_keys=True)) > max_chars

    position = 0
    while _over_budget() and position < len(_SHRINK_ORDER):
        section = facts[_SHRINK_ORDER[position]]
        if section:
            section.pop()
        else:
            position += 1
    return json.dumps(facts, sort_keys=True), facts


def _metrics_supplied(facts: dict[str, Any]) -> tuple[str, ...]:
    """Flatten scalar facts into ``"metric: value"`` strings for the report."""
    return tuple(
        f"{key}: {value}"
        for key, value in facts.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _supplied_values(facts: dict[str, Any]) -> list[float]:
    values: list[float] = []

    def _add(raw: Any) -> None:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val = float(raw)
            values.append(val)
            values.append(abs(val))

    for value in facts.values():
        _add(value)
    for product in facts.get("top_products") or []:
        _add(product.get("net_revenue"))
    for point in facts.get("monthly_trend") or []:
        _add(point.get("net_revenue"))
        _add(point.get("orders"))
    for change in facts.get("period_change") or []:
        for key in ("current", "previous", "change", "change_pct"):
            _add(change.get(key))
    for status in facts.get("status_distribution") or []:
        _add(status.get("orders"))
    return values


def _matches_supplied(value: float, supplied: list[float]) -> bool:
    return any(
        abs(value - candidate) <= max(0.01, 0.005 * abs(candidate))
        for candidate in supplied
    )


def find_unverified_numbers(narrative: str, facts: dict[str, Any]) -> tuple[float, ...]:
    """Numbers in the narrative that match no supplied metric.

    Only money-like values are considered (>= 1000 or with a fractional part);
    small integer counts are ubiquitous in prose and would create noise.
    """
    supplied = _supplied_values(facts)
    found: list[float] = []
    for match in _NUMBER_RE.finditer(narrative):
        token = match.group(0).rstrip("%").replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        if value.is_integer() and 1900 <= value <= 2100:
            continue  # year inside period labels such as "2025-01"
        money_like = abs(value) >= 1000 or not value.is_integer()
        if not money_like or _matches_supplied(value, supplied):
            continue
        rounded = round(value, 2)
        if rounded not in found:
            found.append(rounded)
        if len(found) >= _MAX_UNVERIFIED:
            break
    return tuple(found)


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items = [
        str(entry).strip()[:_MAX_STRING_CHARS]
        for entry in value
        if isinstance(entry, str) and entry.strip()
    ]
    return tuple(items[:_MAX_LIST_ITEMS])


def _narrative(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(entry) for entry in value)
    return " ".join(parts)


def interpret_analytics(
    settings: AISettings,
    analytics: AnalyticsResult,
    *,
    provider: ChatProvider | None = None,
) -> AIInsights:
    """Produce the AI interpretation sheet content for one analytics run.

    Never raises: disabled/unconfigured AI yields a ``SKIPPED`` placeholder and
    provider or parsing failures yield a ``FAILED`` status with the reason, so
    the workbook can always be written.
    """
    logger = get_logger("ai.interpreter")
    reason = settings.unavailable_reason
    if reason is not None:
        logger.info("ai interpretation skipped: %s", reason)
        return AIInsights.disabled(reason)

    facts_json, facts = build_facts_digest(analytics, settings.max_input_chars)
    supplied = _metrics_supplied(facts)
    try:
        active = provider if provider is not None else build_provider(settings)
        raw = active.complete(system=SYSTEM_PROMPT, user=f"{FACTS_MARKER}\n{facts_json}")
        payload = parse_json_response(raw)
        if not isinstance(payload, dict):
            raise AIResponseError("the interpretation response must be a JSON object")
        insights = AIInsights(
            status=AIStatus.GENERATED,
            status_detail="interpretation generated from the deterministic KPI output",
            provider_label=settings.provider_label,
            model=settings.model,
            generated_at=_utc_now(),
            executive_summary=_as_text(payload.get("executive_summary")),
            positive_trends=_as_str_list(payload.get("positive_trends")),
            negative_trends=_as_str_list(payload.get("negative_trends")),
            watch_items=_as_str_list(payload.get("watch_items")),
            management_summary=_as_text(payload.get("management_summary")),
            unverified_numbers=find_unverified_numbers(_narrative(payload), facts),
            metrics_supplied=supplied,
            is_ai_generated=True,
        )
        logger.info(
            "ai interpretation status=%s unverified=%d",
            insights.status.value,
            len(insights.unverified_numbers),
        )
        return insights
    except (AIConfigurationError, AIProviderError, AIResponseError) as exc:
        logger.warning("ai interpretation failed: %s", exc)
        return AIInsights(
            status=AIStatus.FAILED,
            status_detail=f"{type(exc).__name__}: {exc}",
            provider_label=settings.provider_label,
            model=settings.model,
            generated_at=_utc_now(),
            metrics_supplied=supplied,
            is_ai_generated=True,
        )


__all__ = [
    "SYSTEM_PROMPT",
    "build_facts_digest",
    "find_unverified_numbers",
    "interpret_analytics",
]