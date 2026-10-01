"""AI interpretation layer: provider client, interpretation and classification.

The layer is optional and fully configurable (``AI_ENABLED``, ``AI_PROVIDER``):
it only ever *interprets* numbers produced by the deterministic KPI engine and
never changes data. The ``mock`` provider is a deterministic offline stand-in
used for demos and tests.
"""

from __future__ import annotations

from src.ai.classifier import classify_feedback
from src.ai.interpreter import build_facts_digest, interpret_analytics
from src.ai.models import (
    AI_GENERATED_NOTICE,
    AIInsights,
    AIStatus,
    ClassificationRecord,
    ClassificationSummary,
)
from src.ai.provider import ChatProvider, MockProvider, OpenAICompatibleProvider, build_provider

__all__ = [
    "AI_GENERATED_NOTICE",
    "AIInsights",
    "AIStatus",
    "ChatProvider",
    "ClassificationRecord",
    "ClassificationSummary",
    "MockProvider",
    "OpenAICompatibleProvider",
    "build_facts_digest",
    "build_provider",
    "classify_feedback",
    "interpret_analytics",
]