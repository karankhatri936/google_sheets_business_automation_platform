"""Unit tests for the AI interpretation layer (provider, interpreter, classifier)."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from src.ai.classifier import classify_feedback
from src.ai.interpreter import build_facts_digest, interpret_analytics
from src.ai.models import AIStatus
from src.ai.provider import (
    ChatProvider,
    MockProvider,
    OpenAICompatibleProvider,
    build_provider,
    parse_json_response,
    strip_code_fences,
)
from src.analytics.kpi_engine import compute_analytics
from src.config import load_settings
from src.data.quality import QualityOutcome
from src.utils.errors import AIConfigurationError, AIProviderError, AIResponseError


@pytest.fixture
def analytics(quality: QualityOutcome):
    return compute_analytics(quality.clean_data, source="unit_test")


@pytest.fixture
def ai_settings():
    return load_settings(env={"AI_ENABLED": "true", "AI_PROVIDER": "mock"}).ai


class _FixedProvider(ChatProvider):
    """Returns a canned response for every call."""

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system: str, user: str) -> str:
        return self.response


class _ExplodingProvider(ChatProvider):
    def complete(self, *, system: str, user: str) -> str:
        raise AIProviderError("provider unreachable")


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
def test_mock_interpretation_is_generated(ai_settings, analytics):
    insights = interpret_analytics(ai_settings, analytics)
    assert insights.status is AIStatus.GENERATED
    assert insights.has_content
    assert insights.executive_summary
    assert insights.metrics_supplied
    assert insights.unverified_numbers == ()
    assert insights.is_ai_generated


def test_disabled_ai_returns_placeholder(analytics):
    settings = load_settings(env={"AI_ENABLED": "false"}).ai
    insights = interpret_analytics(settings, analytics)
    assert insights.status is AIStatus.SKIPPED
    assert "AI_ENABLED=false" in insights.status_detail
    assert not insights.has_content


def test_provider_failure_marks_failed(ai_settings, analytics):
    insights = interpret_analytics(ai_settings, analytics, provider=_ExplodingProvider())
    assert insights.status is AIStatus.FAILED
    assert "AIProviderError" in insights.status_detail


def test_invalid_json_marks_failed(ai_settings, analytics):
    insights = interpret_analytics(
        ai_settings, analytics, provider=_FixedProvider("not json at all")
    )
    assert insights.status is AIStatus.FAILED
    assert "AIResponseError" in insights.status_detail


def test_unverified_numbers_are_reported(ai_settings, analytics):
    payload = json.dumps(
        {
            "executive_summary": "Revenue hit 999999.99 last quarter.",
            "positive_trends": [],
            "negative_trends": [],
            "watch_items": [],
            "management_summary": "",
        }
    )
    insights = interpret_analytics(ai_settings, analytics, provider=_FixedProvider(payload))
    assert insights.status is AIStatus.GENERATED
    assert 999999.99 in insights.unverified_numbers


def test_facts_digest_shrinks_to_budget(analytics):
    roomy, roomy_facts = build_facts_digest(analytics, 12000)
    tight, tight_facts = build_facts_digest(analytics, 500)
    assert json.loads(roomy)
    assert json.loads(tight)
    assert roomy_facts["monthly_trend"]
    assert tight_facts["monthly_trend"] == []
    assert "net_revenue" in tight_facts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_mock_classification_labels_allowed_categories(ai_settings, quality):
    summary = classify_feedback(ai_settings, quality.clean_data)
    assert summary.status is AIStatus.GENERATED
    assert summary.records
    allowed = set(ai_settings.classification_categories)
    assert {record.category for record in summary.records} <= allowed
    assert sum(count for _, count in summary.counts) == len(summary.records)
    assert summary.unclassified == 0


def test_classification_disabled_is_skipped(quality):
    settings = load_settings(
        env={"AI_PROVIDER": "mock", "AI_CLASSIFICATION_ENABLED": "false"}
    ).ai
    summary = classify_feedback(settings, quality.clean_data)
    assert summary.status is AIStatus.SKIPPED
    assert "AI_CLASSIFICATION_ENABLED" in summary.status_detail


def test_classification_missing_column_is_skipped(quality):
    settings = load_settings(
        env={"AI_PROVIDER": "mock", "AI_CLASSIFICATION_TEXT_COLUMN": "review"}
    ).ai
    summary = classify_feedback(settings, quality.clean_data)
    assert summary.status is AIStatus.SKIPPED
    assert "review" in summary.status_detail


def test_classification_failure_marks_failed(ai_settings, quality):
    summary = classify_feedback(
        ai_settings, quality.clean_data, provider=_ExplodingProvider()
    )
    assert summary.status is AIStatus.FAILED
    assert "AIProviderError" in summary.status_detail


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (urlopen stubbed - no network in tests)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _openai_settings():
    return load_settings(
        env={
            "AI_PROVIDER": "openai_compatible",
            "AI_ENABLED": "true",
            "OPENAI_API_KEY": "sk-test-key",
            "AI_MODEL": "unit-model",
        }
    ).ai


def test_openai_provider_posts_chat_completion(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(_openai_settings())
    text = provider.complete(system="sys", user="usr")

    assert text == "hello"
    request = captured["request"]
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    headers = {key.lower(): value for key, value in request.headers.items()}
    headers.update(
        {key.lower(): value for key, value in request.unredirected_hdrs.items()}
    )
    assert headers["authorization"] == "Bearer sk-test-key"
    body = json.loads(request.data)
    assert body["model"] == "unit-model"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "usr"}
    assert captured["timeout"] is not None


def test_openai_provider_maps_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "server error",
            None,
            io.BytesIO(b'{"error": {"message": "quota exceeded"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(_openai_settings())
    with pytest.raises(AIProviderError, match="HTTP 500"):
        provider.complete(system="s", user="u")


def test_openai_provider_requires_api_key():
    settings = load_settings(
        env={"AI_PROVIDER": "openai_compatible", "AI_ENABLED": "true"}
    ).ai
    with pytest.raises(AIConfigurationError):
        OpenAICompatibleProvider(settings)


def test_build_provider_rejects_disabled_ai():
    settings = load_settings(env={"AI_ENABLED": "false"}).ai
    with pytest.raises(AIConfigurationError):
        build_provider(settings)


def test_build_provider_returns_mock(ai_settings):
    assert isinstance(build_provider(ai_settings), MockProvider)


def test_parse_json_response_strips_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert strip_code_fences(" plain ") == "plain"
    with pytest.raises(AIResponseError):
        parse_json_response("nope")