"""OpenRouter / OpenAI-compatible provider tests (no network, no real API key).

Regression context for the original bug: the provider used to collapse *every*
non-plain-string or empty ``message.content`` into the misleading error
"the AI provider returned an empty message", which hid the real cause. With
OpenRouter's default-on reasoning models the realistic cause is
``content=""`` with ``finish_reason="length"`` (reasoning tokens consumed the
``max_tokens`` budget), so these tests pin both the parsing behaviour and the
diagnostics.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from src.ai.interpreter import interpret_analytics
from src.ai.models import AIStatus
from src.ai.provider import (
    MockProvider,
    OpenAICompatibleProvider,
    build_provider,
    parse_json_response,
)
from src.config import load_settings, normalize_base_url
from src.utils.errors import AIConfigurationError, AIProviderError, AIResponseError

# A clearly fake placeholder - never a real credential.
TEST_API_KEY = "sk-or-v1-unit-test-placeholder-key"

OPENROUTER_ENV = {
    "AI_ENABLED": "true",
    "AI_PROVIDER": "openai_compatible",
    "AI_BASE_URL": "https://openrouter.ai/api/v1",
    "AI_MODEL": "qwen/qwen3.8-27b:free",
    "OPENAI_API_KEY": TEST_API_KEY,
    "AI_MAX_OUTPUT_TOKENS": "900",
    "AI_CLASSIFICATION_ENABLED": "false",
}


def _settings(**overrides: str):
    return load_settings(env={**OPENROUTER_ENV, **overrides}).ai


class _FakeHTTPResponse:
    """Minimal stand-in for ``http.client.HTTPResponse``."""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self._body = payload if isinstance(payload, str) else json.dumps(payload)
        self.status = status
        self.code = status

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _completion(
    content: Any,
    *,
    finish_reason: str = "stop",
    native_finish_reason: str | None = None,
    **extra: Any,
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(extra.pop("message", {}))
    choice: dict[str, Any] = {"index": 0, "message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    if native_finish_reason is not None:
        choice["native_finish_reason"] = native_finish_reason
    payload: dict[str, Any] = {"id": "gen-test", "choices": [choice], "model": "test-model"}
    payload.update(extra)
    return payload


def _install_urlopen(monkeypatch: pytest.MonkeyPatch, responder):
    """Replace ``urlopen`` with ``responder`` and capture the outgoing request."""
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return responder(request)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def _http_error(request, code: int, body: Any):
    raw = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(
        request.full_url, code, "provider error", None, io.BytesIO(raw.encode("utf-8"))
    )


def _headers_of(request) -> dict[str, str]:
    headers = {key.lower(): value for key, value in request.headers.items()}
    headers.update({key.lower(): value for key, value in request.unredirected_hdrs.items()})
    return headers


def _provider(settings=None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(settings or _settings())



# ---------------------------------------------------------------------------
# Configuration / endpoint construction
# ---------------------------------------------------------------------------
def test_openrouter_endpoint_is_constructed_once():
    assert _provider().endpoint == "https://openrouter.ai/api/v1/chat/completions"


def test_endpoint_tolerates_trailing_slash_and_full_url():
    for settings in (
        _settings(AI_BASE_URL="https://openrouter.ai/api/v1/"),
        _settings(AI_BASE_URL="https://openrouter.ai/api/v1/chat/completions"),
    ):
        endpoint = _provider(settings).endpoint
        assert endpoint == "https://openrouter.ai/api/v1/chat/completions"
        assert "/v1/v1" not in endpoint
        assert endpoint.count("/chat/completions") == 1


def test_normalize_base_url_rejects_relative_values():
    with pytest.raises(Exception) as excinfo:
        normalize_base_url("openrouter.ai/api/v1")
    assert "absolute http(s) URL" in str(excinfo.value)


def test_invalid_base_url_is_a_configuration_error():
    for bad_value in ("openrouter.ai/api/v1", "ftp://openrouter.ai/api/v1"):
        with pytest.raises(Exception) as excinfo:
            load_settings(
                env={
                    "AI_PROVIDER": "openai_compatible",
                    "AI_ENABLED": "true",
                    "AI_BASE_URL": bad_value,
                    "AI_MODEL": "qwen/qwen3.8-27b:free",
                }
            )
        assert "AI_BASE_URL" in str(excinfo.value)


def test_missing_base_url_falls_back_to_the_documented_default():
    settings = load_settings(
        env={
            "AI_PROVIDER": "openai_compatible",
            "AI_ENABLED": "true",
            "AI_MODEL": "gpt-4o-mini",
            "OPENAI_API_KEY": TEST_API_KEY,
        }
    ).ai
    assert settings.base_url == "https://api.openai.com/v1"
    assert _provider(settings).endpoint == "https://api.openai.com/v1/chat/completions"


def test_openrouter_model_requires_vendor_prefix():
    with pytest.raises(Exception) as excinfo:
        load_settings(
            env={
                "AI_PROVIDER": "openai_compatible",
                "AI_ENABLED": "true",
                "AI_BASE_URL": "https://openrouter.ai/api/v1",
                "AI_MODEL": "qwen3.8-27b:free",
            }
        )
    assert "vendor" in str(excinfo.value)


def test_non_openrouter_base_url_keeps_existing_behaviour():
    settings = load_settings(
        env={
            "AI_PROVIDER": "openai_compatible",
            "AI_ENABLED": "true",
            "AI_BASE_URL": "https://api.openai.com/v1",
            "AI_MODEL": "gpt-4o-mini",
        }
    ).ai
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "https://api.openai.com/v1"


def test_missing_api_key_is_a_clear_configuration_error():
    with pytest.raises(AIConfigurationError) as excinfo:
        OpenAICompatibleProvider(_settings(OPENAI_API_KEY=""))
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message and "never commit it" in message


# ---------------------------------------------------------------------------
# Request construction (headers / body / model selection)
# ---------------------------------------------------------------------------
def test_request_carries_bearer_auth_and_configured_model(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    provider = _provider()
    assert provider.complete(system="sys", user="usr") == "ok"

    request = captured["request"]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    headers = _headers_of(request)
    assert headers["authorization"] == f"Bearer {TEST_API_KEY}"
    assert headers["content-type"] == "application/json"
    assert "http-referer" not in headers and "x-title" not in headers

    body = json.loads(request.data)
    assert body["model"] == "qwen/qwen3.8-27b:free"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["stream"] is False
    assert captured["timeout"] == provider.settings.timeout_seconds


def test_optional_openrouter_attribution_headers_are_sent_when_configured(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    provider = _provider(
        _settings(
            AI_HTTP_REFERER="https://github.com/example/gsba",
            AI_APP_TITLE="Google Sheets Business Automation",
        )
    )
    provider.complete(system="s", user="u")
    headers = _headers_of(captured["request"])
    assert headers["http-referer"] == "https://github.com/example/gsba"
    assert headers["x-title"] == "Google Sheets Business Automation"


def test_secret_never_appears_in_logs_or_errors(monkeypatch, caplog):
    """The API key must not leak through logs or exception messages."""
    with caplog.at_level("DEBUG"):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeHTTPResponse(_completion("hello")),
        )
        _provider().complete(system="s", user="u")

    def raise_401(request, timeout=None):
        raise _http_error(
            request,
            401,
            {"error": {"message": f"invalid credentials near Bearer {TEST_API_KEY}", "code": 401}},
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert TEST_API_KEY not in logged
    assert TEST_API_KEY not in str(excinfo.value)
    assert "***" in str(excinfo.value)




# ---------------------------------------------------------------------------
# Response parsing (defensive)
# ---------------------------------------------------------------------------
def test_plain_string_content_is_returned(monkeypatch):
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(_completion("hello")))
    assert _provider().complete(system="s", user="u") == "hello"


def test_content_given_as_text_parts_is_flattened(monkeypatch):
    parts = [
        {"type": "text", "text": "revenue is up"},
        {"type": "text", "text": "watch refunds"},
        {"type": "image_url", "image_url": {"url": "https://example.com/i.png"}},
        "trailing plain string",
    ]
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(_completion(parts)))
    assert _provider().complete(system="s", user="u") == (
        "revenue is up\nwatch refunds\ntrailing plain string"
    )


def test_empty_content_reports_actionable_diagnostics(monkeypatch, caplog):
    """The original bug: this used to be a bare 'empty message' error."""
    payload = _completion(
        "",
        finish_reason="length",
        message={"reasoning": "thinking out loud..."},
        usage={"prompt_tokens": 1200, "completion_tokens": 900, "total_tokens": 2100},
    )
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))

    with caplog.at_level("INFO"):
        with pytest.raises(AIResponseError) as excinfo:
            _provider().complete(system="s", user="u")

    message = str(excinfo.value)
    assert "no usable message content" in message
    assert "finish_reason='length'" in message
    assert "http_status=200" in message
    assert "provider=openai_compatible:qwen/qwen3.8-27b:free" in message
    assert "reasoning_present=True" in message
    assert "completion_tokens=900" in message
    assert "AI_MAX_OUTPUT_TOKENS" in message


def test_null_content_without_length_hint_explains_content_type(monkeypatch):
    payload = _completion(None, finish_reason="stop")
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))

    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    message = str(excinfo.value)
    assert "no usable message content" in message
    assert "content_type=NoneType" in message
    assert "usage=unavailable" in message


def test_reasoning_only_reply_suggests_disabling_reasoning(monkeypatch):
    payload = _completion(
        "", finish_reason="stop", message={"reasoning_details": [{"type": "text"}]}
    )
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))

    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "only reasoning tokens" in str(excinfo.value)


def test_missing_choices_is_reported(monkeypatch):
    _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse({"id": "gen-test", "model": "m"})
    )
    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "no choices" in str(excinfo.value)
    assert "top_level_keys=" in str(excinfo.value)


def test_empty_choices_is_reported(monkeypatch):
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse({"choices": []}))
    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "no choices" in str(excinfo.value)


def test_missing_message_is_reported(monkeypatch):
    _install_urlopen(
        monkeypatch,
        lambda request: _FakeHTTPResponse({"choices": [{"index": 0, "finish_reason": "stop"}]}),
    )
    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "no message object" in str(excinfo.value)
    assert "choice_keys=" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Transport / API error handling
# ---------------------------------------------------------------------------
def test_http_4xx_surfaces_status_and_provider_message(monkeypatch):
    def raise_401(request, timeout=None):
        raise _http_error(
            request,
            401,
            {"error": {"message": "No auth credentials found", "code": 401, "type": "auth"}},
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")

    message = str(excinfo.value)
    assert "HTTP 401" in message
    assert "No auth credentials found" in message
    assert "code=401" in message
    assert "qwen/qwen3.8-27b:free" in message


def test_http_5xx_surfaces_status(monkeypatch):
    def raise_503(request, timeout=None):
        raise _http_error(
            request, 503, {"error": {"message": "no available provider", "code": 503}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_503)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "HTTP 503" in str(excinfo.value)
    assert "no available provider" in str(excinfo.value)


def test_http_error_without_body_still_reports_status(monkeypatch):
    def raise_400(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "bad request", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_400)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "HTTP 400" in str(excinfo.value)


def test_error_body_with_http_200_raises_provider_error(monkeypatch):
    _install_urlopen(
        monkeypatch,
        lambda request: _FakeHTTPResponse(
            {"error": {"message": "model not found", "code": 404}}
        ),
    )
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "model not found" in str(excinfo.value)


def test_invalid_json_response_is_reported(monkeypatch):
    _install_urlopen(
        monkeypatch,
        lambda request: _FakeHTTPResponse("<html>gateway timeout</html>"),
    )
    with pytest.raises(AIResponseError) as excinfo:
        _provider().complete(system="s", user="u")
    message = str(excinfo.value)
    assert "non-JSON response" in message
    assert "gateway timeout" in message


def test_unreachable_provider_is_reported(monkeypatch):
    def raise_url_error(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_error)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "could not reach the AI provider" in str(excinfo.value)
    assert "connection refused" in str(excinfo.value)



# ---------------------------------------------------------------------------
# Interpreter integration and offline mock behaviour
# ---------------------------------------------------------------------------
@pytest.fixture
def analytics(quality):
    from src.analytics.kpi_engine import compute_analytics

    return compute_analytics(quality.clean_data, source="openrouter_unit_test")


def test_interpreter_surfaces_empty_content_diagnostics(monkeypatch, analytics, caplog):
    """The reported symptom must now name the real cause."""
    payload = _completion("", finish_reason="length", message={"reasoning": "..."})
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))

    with caplog.at_level("WARNING"):
        insights = interpret_analytics(_settings(), analytics)

    assert insights.status is AIStatus.FAILED
    assert "AIResponseError" in insights.status_detail
    assert "finish_reason='length'" in insights.status_detail
    assert "AI_MAX_OUTPUT_TOKENS" in insights.status_detail
    assert any("no usable message content" in record.getMessage() for record in caplog.records)


def test_interpreter_accepts_a_normal_openrouter_reply(monkeypatch, analytics):
    narrative = json.dumps(
        {
            "executive_summary": f"net revenue reached {round(analytics.overall.net_revenue, 2)}.",
            "positive_trends": ["revenue grew versus the previous period"],
            "negative_trends": [],
            "watch_items": ["refund share is worth monitoring"],
            "management_summary": "keep investing in the strongest product lines",
        }
    )
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion(narrative))
    )

    insights = interpret_analytics(_settings(), analytics)

    assert insights.status is AIStatus.GENERATED
    assert insights.model == "qwen/qwen3.8-27b:free"
    assert insights.provider_label == "openai_compatible:qwen/qwen3.8-27b:free"
    assert insights.unverified_numbers == ()
    assert captured["request"].full_url == "https://openrouter.ai/api/v1/chat/completions"


def test_classification_failure_reports_provider_status(monkeypatch, quality):
    from src.ai.classifier import classify_feedback

    def raise_429(request, timeout=None):
        raise _http_error(
            request, 429, {"error": {"message": "rate limit exceeded", "code": 429}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_429)
    summary = classify_feedback(
        _settings(AI_CLASSIFICATION_ENABLED="true"), quality.clean_data
    )
    assert summary.status is AIStatus.FAILED
    assert "HTTP 429" in summary.status_detail
    assert "rate limit exceeded" in summary.status_detail


def test_mock_provider_still_runs_completely_offline(monkeypatch, analytics, quality):
    from src.ai.classifier import classify_feedback

    calls: list[str] = []

    def forbidden(request, timeout=None):  # pragma: no cover - must never run
        calls.append(request.full_url)
        raise AssertionError("the mock provider must not perform HTTP calls")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    settings = load_settings(
        env={"AI_ENABLED": "true", "AI_PROVIDER": "mock", "AI_CLASSIFICATION_ENABLED": "true"}
    ).ai

    insights = interpret_analytics(settings, analytics)
    summary = classify_feedback(settings, quality.clean_data)

    assert isinstance(build_provider(settings), MockProvider)
    assert insights.status is AIStatus.GENERATED
    assert summary.status is AIStatus.GENERATED
    assert calls == []


def test_build_provider_selects_the_openai_compatible_provider():
    provider = build_provider(_settings())
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.endpoint == "https://openrouter.ai/api/v1/chat/completions"



# ---------------------------------------------------------------------------
# Output budget and reasoning configuration (regression: a 900-token budget was
# consumed entirely by reasoning, producing HTTP 200 + empty content)
# ---------------------------------------------------------------------------
def test_default_output_budget_is_3000_for_interpretation():
    settings = load_settings(
        env={k: v for k, v in OPENROUTER_ENV.items() if k != "AI_MAX_OUTPUT_TOKENS"}
    ).ai
    assert settings.max_output_tokens == 3000


def test_output_budget_is_configurable_through_env(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    provider = _provider(_settings(AI_MAX_OUTPUT_TOKENS="5000"))
    assert provider.settings.max_output_tokens == 5000
    provider.complete(system="s", user="u")
    assert json.loads(captured["request"].data)["max_tokens"] == 5000


def test_reasoning_defaults_to_minimal_for_openrouter(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    provider = _provider(_settings())
    provider.complete(system="s", user="u")
    body = json.loads(captured["request"].data)
    assert provider.settings.reasoning_effort == "minimal"
    assert body["reasoning"] == {"effort": "minimal", "exclude": True}


def test_other_gateways_never_receive_the_reasoning_field(monkeypatch):
    """Non-OpenRouter endpoints (OpenAI, Ollama, ...) must be unchanged."""
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    settings = _settings(AI_BASE_URL="https://api.openai.com/v1", AI_MODEL="gpt-4o-mini")
    assert settings.reasoning_payload is None
    assert settings.send_reasoning_params is False
    _provider(settings).complete(system="s", user="u")
    assert "reasoning" not in json.loads(captured["request"].data)


def test_reasoning_can_be_forced_for_a_compatible_gateway(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    settings = _settings(
        AI_BASE_URL="https://gateway.example.com/v1",
        AI_MODEL="vendor/model",
        AI_SEND_REASONING_PARAMS="true",
    )
    _provider(settings).complete(system="s", user="u")
    assert json.loads(captured["request"].data)["reasoning"]["effort"] == "minimal"


def test_reasoning_can_be_turned_off_completely(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    settings = _settings(AI_SEND_REASONING_PARAMS="false")
    assert settings.reasoning_payload is None
    _provider(settings).complete(system="s", user="u")
    assert "reasoning" not in json.loads(captured["request"].data)


def test_reasoning_effort_auto_omits_the_field(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    settings = _settings(AI_REASONING_EFFORT="auto")
    assert settings.reasoning_effort is None
    _provider(settings).complete(system="s", user="u")
    assert "reasoning" not in json.loads(captured["request"].data)


def test_custom_reasoning_effort_and_budget(monkeypatch):
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion("ok"))
    )
    settings = _settings(AI_REASONING_EFFORT="medium", AI_REASONING_MAX_TOKENS="1024")
    _provider(settings).complete(system="s", user="u")
    assert json.loads(captured["request"].data)["reasoning"] == {
        "effort": "medium",
        "exclude": True,
        "max_tokens": 1024,
    }


def test_reasoning_effort_off_is_an_alias_for_none():
    assert _settings(AI_REASONING_EFFORT="off").reasoning_effort == "none"
    assert _settings(AI_REASONING_EFFORT="disabled").reasoning_effort == "none"


def test_invalid_reasoning_effort_is_rejected():
    with pytest.raises(Exception) as excinfo:
        _settings(AI_REASONING_EFFORT="extreme")
    assert "AI_REASONING_EFFORT" in str(excinfo.value)



def test_reasoning_response_with_visible_content_is_used(monkeypatch):
    payload = _completion(
        "net revenue reached 85410.87 across 47 orders.",
        message={"reasoning": "checking the KPI digest first"},
        usage={"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
    )
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))
    text = _provider().complete(system="s", user="u")
    assert text.startswith("net revenue reached")


def test_reasoning_exhausting_the_token_budget_is_a_failure(monkeypatch):
    """HTTP 200 + finish_reason=length + empty content must never return ""."""
    payload = _completion(
        "",
        finish_reason="length",
        native_finish_reason="length",
        message={"reasoning": "still thinking..."},
        usage={
            "prompt_tokens": 954,
            "completion_tokens": 900,
            "total_tokens": 1854,
            "completion_tokens_details": {"reasoning_tokens": 900},
        },
    )
    _install_urlopen(monkeypatch, lambda request: _FakeHTTPResponse(payload))

    with pytest.raises(AIResponseError) as excinfo:
        _provider(_settings(AI_MAX_OUTPUT_TOKENS="900")).complete(system="s", user="u")

    message = str(excinfo.value)
    assert "no usable message content" in message
    assert "finish_reason='length'" in message
    assert "native_finish_reason='length'" in message
    assert "reasoning_tokens=900" in message
    assert "AI_MAX_OUTPUT_TOKENS (currently 900)" in message
    assert "AI_REASONING_EFFORT=minimal" in message


def test_mock_provider_ignores_reasoning_settings_and_stays_offline(
    monkeypatch, analytics, quality
):
    from src.ai.classifier import classify_feedback

    def forbidden(request, timeout=None):  # pragma: no cover - must never run
        raise AssertionError("the mock provider must not perform HTTP calls")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    settings = load_settings(
        env={
            "AI_ENABLED": "true",
            "AI_PROVIDER": "mock",
            "AI_REASONING_EFFORT": "xhigh",
            "AI_MAX_OUTPUT_TOKENS": "150",
            "AI_CLASSIFICATION_ENABLED": "true",
        }
    ).ai

    insights = interpret_analytics(settings, analytics)
    summary = classify_feedback(settings, quality.clean_data)

    assert insights.status is AIStatus.GENERATED
    assert insights.has_content
    assert summary.status is AIStatus.GENERATED



# ---------------------------------------------------------------------------
# Tolerant response parsing (free/aggregated models add preambles and fences)
# ---------------------------------------------------------------------------
def test_parse_json_response_handles_preamble_and_trailing_text():
    assert parse_json_response('Here is the JSON:\n{"a": 1}\nHope this helps!') == {"a": 1}


def test_parse_json_response_handles_fenced_json_with_preamble():
    text = 'Sure! Here is the interpretation:\n```json\n{"executive_summary": "up"}\n```\n'
    assert parse_json_response(text) == {"executive_summary": "up"}


def test_parse_json_response_handles_trailing_commas_and_arrays():
    assert parse_json_response('[{"index": 0, "category": "positive"},]') == [
        {"index": 0, "category": "positive"}
    ]
    assert parse_json_response('result: {"a": [1, 2,],}') == {"a": [1, 2]}


def test_parse_json_response_rejects_non_json_without_echoing_it():
    secret_ish = "The customer wrote something private about order 12345"
    with pytest.raises(AIResponseError) as excinfo:
        parse_json_response(secret_ish)
    message = str(excinfo.value)
    assert "is not valid JSON" in message
    assert "12345" not in message and "private" not in message
    assert "char(s) starting with letter" in message


def test_interpreter_accepts_a_json_reply_wrapped_in_prose(monkeypatch, analytics):
    wrapped = (
        "Sure, here is the interpretation you asked for:\n```json\n"
        + json.dumps(
            {
                "executive_summary": "net revenue was 85410.87",
                "positive_trends": [],
                "negative_trends": [],
                "watch_items": [],
                "management_summary": "",
            }
        )
        + "\n```\nLet me know if you need more detail."
    )
    captured = _install_urlopen(
        monkeypatch, lambda request: _FakeHTTPResponse(_completion(wrapped))
    )

    insights = interpret_analytics(_settings(), analytics)

    assert captured["request"].full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert insights.status is AIStatus.GENERATED
    assert insights.executive_summary
    assert insights.unverified_numbers == ()



# ---------------------------------------------------------------------------
# Rejected reasoning parameters (real OpenRouter response: HTTP 400
# "Reasoning is mandatory for this endpoint and cannot be disabled.")
# ---------------------------------------------------------------------------
MANDATORY_REASONING_ERROR = {
    "error": {
        "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
        "code": 400,
        "metadata": {"provider_name": None},
    }
}


def test_provider_retries_once_without_reasoning_when_rejected(monkeypatch):
    bodies: list[dict] = []

    def responder(request, timeout=None):
        bodies.append(json.loads(request.data))
        if len(bodies) == 1:
            raise _http_error(request, 400, MANDATORY_REASONING_ERROR)
        return _FakeHTTPResponse(_completion("visible answer"))

    monkeypatch.setattr(urllib.request, "urlopen", responder)

    assert _provider().complete(system="s", user="u") == "visible answer"
    assert len(bodies) == 2
    assert bodies[0]["reasoning"] == {"effort": "minimal", "exclude": True}
    assert "reasoning" not in bodies[1]
    assert bodies[1]["max_tokens"] == 900  # the configured budget is untouched


def test_rejected_reasoning_without_recovery_still_fails(monkeypatch):
    def always_400(request, timeout=None):
        raise _http_error(request, 400, MANDATORY_REASONING_ERROR)

    monkeypatch.setattr(urllib.request, "urlopen", always_400)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "HTTP 400" in str(excinfo.value)


def test_unrelated_provider_error_is_not_retried(monkeypatch):
    calls: list[str] = []

    def rate_limited(request, timeout=None):
        calls.append(request.full_url)
        raise _http_error(request, 429, {"error": {"message": "rate limit exceeded"}})

    monkeypatch.setattr(urllib.request, "urlopen", rate_limited)
    with pytest.raises(AIProviderError) as excinfo:
        _provider().complete(system="s", user="u")
    assert "rate limit exceeded" in str(excinfo.value)
    assert len(calls) == 1


def test_no_retry_when_reasoning_is_not_sent(monkeypatch):
    calls: list[str] = []

    def always_400(request, timeout=None):
        calls.append(request.full_url)
        raise _http_error(request, 400, MANDATORY_REASONING_ERROR)

    monkeypatch.setattr(urllib.request, "urlopen", always_400)
    with pytest.raises(AIProviderError):
        _provider(_settings(AI_SEND_REASONING_PARAMS="false")).complete(system="s", user="u")
    assert len(calls) == 1

