"""Chat-completions provider for the AI interpretation layer.

The provider is intentionally thin: it sends one system + user message pair and
returns the assistant's text. Prompt design and response parsing live in the
interpretation/classification modules.

Works with any OpenAI-compatible ``/chat/completions`` endpoint, including
aggregators such as OpenRouter (``AI_BASE_URL=https://openrouter.ai/api/v1``,
``AI_MODEL=<vendor>/<model>[:variant]``). The model is always taken from
configuration - never hard-coded here.

No HTTP client dependency on purpose: the standard library ``urllib`` is used
so the dependency surface stays small (see requirements.txt).

Response handling is defensive because aggregators do not all reply in exactly
the same shape: ``choices``/``message`` are validated, ``message.content`` may
be a string or a list of text parts, provider error payloads are surfaced with
their HTTP status and message, and a genuinely unusable reply raises
``AIResponseError`` with the diagnostics needed to fix it (finish reason,
content type, reasoning/usage hints) instead of a generic "empty message".

Secrets never reach the logs: outgoing ``Authorization`` headers are never
logged and every provider/API string is passed through :meth:`_redact`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from src.config.logging_config import get_logger
from src.config.settings import AISettings, normalize_base_url
from src.utils.errors import AIConfigurationError, AIProviderError, AIResponseError

CHAT_COMPLETIONS_PATH = "/chat/completions"

# Prompt markers shared with the mock provider (and the interpreter/classifier).
FACTS_MARKER = "FACTS_JSON:"
ITEMS_MARKER = "ITEMS_JSON:"
CATEGORIES_MARKER = "CATEGORIES_JSON:"

_FENCE_RE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# Guards used by :meth:`OpenAICompatibleProvider._redact` so no credential can
# leak into an exception message or a log line.
_AUTH_HEADER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{4,}")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9._\-]{6,}\b")

_MAX_DIAGNOSTIC_CHARS = 300

# Providers reject reasoning parameters in a few different wordings; all of them
# name the field, so a 4xx mentioning "reasoning" is treated as a rejected field.
_REASONING_REJECTION_RE = re.compile(
    r"(?i)\breasoning\b[^.]*\b(mandatory|unsupported|not supported|invalid|cannot|rejected)"
)


class ChatProvider:
    """Minimal chat-completions interface used by the AI layer."""

    def complete(self, *, system: str, user: str) -> str:
        raise NotImplementedError


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from a model response (many models add them)."""
    fenced = _FENCE_RE.fullmatch(text)
    return fenced.group(1) if fenced else text.strip()


def _loads_tolerant(payload: str) -> Any | None:
    """``json.loads`` that also tolerates trailing commas before a bracket."""
    for candidate in (payload, _TRAILING_COMMA_RE.sub(r"\1", payload)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_first_json(payload: str) -> Any | None:
    """Decode the first JSON object/array found anywhere in ``payload``.

    Free/aggregated models often prepend a sentence ("Here is the JSON:") or
    append commentary; ``raw_decode`` stops at the end of the first valid value
    so both cases parse. ``None`` means no JSON value was found at all.
    """
    decoder = json.JSONDecoder()
    canonical = _TRAILING_COMMA_RE.sub(r"\1", payload)
    for text in (payload, canonical):
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return value
    return None


def _payload_shape(payload: str) -> str:
    """Safe description of an unparseable payload (never its contents)."""
    stripped = payload.strip()
    if not stripped:
        return "empty response"
    head = stripped[0]
    kind = "letter" if head.isalpha() else ("digit" if head.isdigit() else repr(head))
    return f"{len(stripped)} char(s) starting with {kind}"


def parse_json_response(text: str) -> Any:
    """Parse a JSON payload out of a model response.

    Handles the shapes real models produce in practice: a bare JSON value,
    Markdown-fenced JSON, JSON preceded by a short preamble or followed by
    commentary, and JSON with trailing commas. A response containing no JSON at
    all raises :class:`AIResponseError` describing only its *shape* (length and
    first character) so no user or provider content is copied into logs.
    """
    payload = strip_code_fences(text)
    parsed = _loads_tolerant(payload)
    if parsed is not None:
        return parsed
    extracted = _extract_first_json(payload)
    if extracted is not None:
        return extracted
    raise AIResponseError(f"the AI response is not valid JSON ({_payload_shape(payload)})")


def _marker_payload(marker: str, text: str) -> str | None:
    """Return everything after ``marker`` (may span multiple lines), or None."""
    index = text.find(marker)
    if index < 0:
        return None
    return text[index + len(marker) :].strip()


class OpenAICompatibleProvider(ChatProvider):
    """POSTs to any OpenAI-compatible ``/chat/completions`` endpoint via urllib.

    The base URL is normalised through
    :func:`src.config.settings.normalize_base_url` so a configured value such as
    ``https://openrouter.ai/api/v1`` (or even the full
    ``.../v1/chat/completions`` URL) always resolves to exactly one endpoint.
    """

    def __init__(self, settings: AISettings) -> None:
        if not settings.api_key:
            raise AIConfigurationError(
                f"the {settings.provider} provider at {settings.base_url} requires an API key "
                f"in environment variable {settings.api_key_env_var} "
                "(set it in .env or the process environment; never commit it)"
            )
        self.settings = settings
        self._logger = get_logger("ai.provider")

    @property
    def endpoint(self) -> str:
        """Fully qualified chat-completions URL (never doubled)."""
        return f"{normalize_base_url(self.settings.base_url)}{CHAT_COMPLETIONS_PATH}"

    def _headers(self) -> dict[str, str]:
        """Request headers. The API key is added here and never logged."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        # Optional OpenRouter attribution headers - only sent when configured.
        if self.settings.http_referer:
            headers["HTTP-Referer"] = self.settings.http_referer
        if self.settings.app_title:
            headers["X-Title"] = self.settings.app_title
        return headers

    def _body(
        self, system: str, user: str, reasoning: dict[str, object] | None
    ) -> bytes:
        """Non-streaming chat-completions payload built from existing prompts.

        ``reasoning`` is ``None`` when the field must be omitted: either it is
        not supported by the configured target, or the provider rejected it.
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
            "stream": False,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return json.dumps(payload).encode("utf-8")

    def complete(self, *, system: str, user: str) -> str:
        """Send one non-streaming chat completion and return the assistant text.

        Reasoning parameters are sent when the configuration supports them (see
        :attr:`AISettings.reasoning_payload`). If the provider rejects that field
        (for example an endpoint where reasoning is mandatory rejects
        ``effort: "none"``) the request is retried exactly once without it, so
        one unsuitable model cannot break the whole AI feature.
        """
        reasoning = self.settings.reasoning_payload
        try:
            return self._send(system, user, reasoning)
        except AIProviderError as exc:
            if reasoning is None or not _REASONING_REJECTION_RE.search(str(exc)):
                raise
            self._logger.warning(
                "the provider rejected the reasoning parameters for model %s "
                "(%s); retrying once without them",
                self.settings.model,
                exc,
            )
            return self._send(system, user, None)

    def _send(
        self, system: str, user: str, reasoning: dict[str, object] | None
    ) -> str:
        request = urllib.request.Request(
            self.endpoint,
            data=self._body(system, user, reasoning),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8", "replace")
                status = getattr(response, "status", None) or getattr(response, "code", 200)
        except urllib.error.HTTPError as exc:
            raise AIProviderError(self._http_error_message(exc)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AIProviderError(
                self._redact(
                    f"could not reach the AI provider at {self.endpoint} "
                    f"({self.settings.provider_label}): {exc}"
                )
            ) from exc

        if isinstance(status, int) and not 200 <= status < 300:
            raise AIProviderError(
                f"the AI provider returned HTTP {status} for model "
                f"{self.settings.model!r} without a usable body"
            )
        return self._extract_message_text(raw, status)

    # -- diagnostics ---------------------------------------------------------
    def _context(self, status: int | None) -> str:
        """Safe request metadata used for every diagnostic message."""
        return (
            f"provider={self.settings.provider_label}; "
            f"endpoint={self.endpoint}; http_status={status}"
        )

    def _http_error_message(self, exc: urllib.error.HTTPError) -> str:
        """Turn an HTTP failure into an actionable, secret-free message."""
        body = ""
        if exc.fp is not None:
            try:
                body = exc.read().decode("utf-8", "replace")
            except (OSError, ValueError):  # pragma: no cover - defensive
                body = ""
        detail = self._provider_error_detail(body) or (exc.reason and str(exc.reason)) or "no body"
        return self._redact(
            f"the AI provider returned HTTP {exc.code} for model "
            f"{self.settings.model!r}: {detail} ({self._context(exc.code)})"
        )

    @staticmethod
    def _provider_error_detail(body: str) -> str | None:
        """Extract a readable ``error`` message from a provider payload."""
        collapsed = " ".join(body.split())
        if not collapsed:
            return None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return collapsed[:_MAX_DIAGNOSTIC_CHARS]
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return collapsed[:_MAX_DIAGNOSTIC_CHARS]
        parts = [str(error.get("message") or "provider reported an error").strip()]
        if error.get("code") is not None:
            parts.append(f"code={error['code']}")
        if error.get("type"):
            parts.append(f"type={error['type']}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict) and metadata:
            parts.append(f"metadata={json.dumps(metadata)[:_MAX_DIAGNOSTIC_CHARS]}")
        return " | ".join(parts)

    def _redact(self, text: str) -> str:
        """Strip anything credential-shaped before it reaches a log or message."""
        secret = self.settings.api_key
        if secret:
            text = text.replace(secret, "***")
        text = _AUTH_HEADER_RE.sub("Bearer ***", text)
        return _API_KEY_RE.sub("***", text)

    def _extract_message_text(self, raw: str, status: int | None = None) -> str:
        """Validate a chat-completions payload and return the assistant text.

        Raises
        ------
        AIProviderError
            When the payload is a 200-with-``error`` body or unusable for a
            provider side reason.
        AIResponseError
            When the payload is not JSON, has no ``choices``/``message``, or
            carries no usable text. The message always explains *what* was
            received (finish reason, content type, usage) instead of reporting a
            generic empty message.
        """
        context = self._context(status)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            snippet = " ".join(raw.split())[:_MAX_DIAGNOSTIC_CHARS] or "<empty body>"
            raise AIResponseError(
                self._redact(f"the AI provider returned a non-JSON response ({context}): {snippet}")
            ) from exc

        if not isinstance(payload, dict):
            raise AIResponseError(
                self._redact(
                    f"the AI provider response is not a JSON object ({context}; "
                    f"received {type(payload).__name__})"
                )
            )

        # Some gateways answer HTTP 200 with an {"error": {...}} body.
        if payload.get("error") is not None:
            detail = self._provider_error_detail(json.dumps({"error": payload["error"]}))
            raise AIProviderError(
                self._redact(
                    f"the AI provider reported an error ({context}): {detail or 'unknown error'}"
                )
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIResponseError(
                self._redact(
                    f"the AI provider response has no choices ({context}; "
                    f"top_level_keys={sorted(payload)[:10]})"
                )
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise AIResponseError(
                self._redact(
                    f"the AI provider response has a malformed choice ({context}; "
                    f"choice_type={type(choice).__name__})"
                )
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise AIResponseError(
                self._redact(
                    f"the AI provider response has no message object ({context}; "
                    f"choice_keys={sorted(choice)[:10]})"
                )
            )

        content = message.get("content")
        text = _content_to_text(content)
        finish_reason = choice.get("finish_reason")
        if text.strip():
            self._logger.info(
                "ai provider response ok (provider=%s; model=%s; http_status=%s; "
                "finish_reason=%s; usable_text=True; chars=%d)",
                self.settings.provider,
                self.settings.model,
                status,
                finish_reason,
                len(text),
            )
            return text

        raise AIResponseError(
            self._redact(
                f"the AI provider returned no usable message content ({context}; "
                f"finish_reason={finish_reason!r}; "
                f"native_finish_reason={choice.get('native_finish_reason')!r}; "
                f"content_type={type(content).__name__}; "
                f"content_chars={len(text)}; "
                f"reasoning_present={_has_reasoning(message)}; "
                f"usage={_usage_summary(payload.get('usage'))})"
                f"{_empty_content_hint(self.settings, finish_reason, message)}"
            )
        )


def _content_part_text(part: Any) -> str:
    """Text of a single ``content`` part (strings and OpenAI-style dicts)."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                return _content_to_text(value)
    return ""


def _content_to_text(content: Any) -> str:
    """Flatten ``message.content`` into plain text.

    Handles the plain-string form used by OpenAI/OpenRouter as well as the
    list-of-parts form emitted by some providers
    (``[{"type": "text", "text": "..."}]``). Non-text parts (images, tool calls)
    are ignored; malformed values yield "" so the caller can report exactly what
    was received instead of inventing content.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        return "\n".join(text for text in map(_content_part_text, content) if text)
    if isinstance(content, dict):
        return _content_part_text(content)
    return ""


def _has_reasoning(message: dict[str, Any]) -> bool:
    """True when the provider attached reasoning tokens but no answer text."""
    return bool(message.get("reasoning") or message.get("reasoning_details"))


def _usage_summary(usage: Any) -> str:
    """Token counts only (never prompt/completion text)."""
    if not isinstance(usage, dict):
        return "unavailable"
    parts: list[str] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            parts.append(f"{key}={usage[key]}")
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        parts.append(f"reasoning_tokens={details['reasoning_tokens']}")
    return ", ".join(parts) if parts else "unavailable"


def _empty_content_hint(
    settings: AISettings, finish_reason: Any, message: dict[str, Any]
) -> str:
    """Actionable suffix explaining the most likely cause of empty content."""
    if finish_reason == "length":
        return (
            " - the model used its entire output budget before writing an answer: raise "
            f"AI_MAX_OUTPUT_TOKENS (currently {settings.max_output_tokens}) and/or lower "
            f"thinking with AI_REASONING_EFFORT=minimal (currently "
            f"{settings.reasoning_effort or 'model default'}), because reasoning models "
            "(for example OpenRouter 'qwen/...' slugs) spend output tokens on hidden "
            "reasoning first"
        )
    if _has_reasoning(message):
        return (
            " - the model returned only reasoning tokens; set AI_REASONING_EFFORT=minimal "
            "or a lower effort level for this model"
        )
    if finish_reason == "content_filter":
        return " - the provider filtered the completion (finish_reason=content_filter)"
    return ""


class MockProvider(ChatProvider):
    """Deterministic, offline stand-in used for demos and tests.

    It never invents numbers: every figure in its interpretation is taken from
    the ``FACTS_JSON`` block supplied by the interpreter, and every label comes
    from the allowed category list supplied by the classifier.
    """

    def __init__(self, settings: AISettings) -> None:
        self.settings = settings

    def complete(self, *, system: str, user: str) -> str:
        items_payload = _marker_payload(ITEMS_MARKER, user)
        if items_payload is not None:
            return self._classify(system, items_payload)
        facts_payload = _marker_payload(FACTS_MARKER, user)
        if facts_payload is None:
            raise AIResponseError("the prompt did not contain a FACTS_JSON block")
        return self._interpret(facts_payload)

    def _interpret(self, facts_payload: str) -> str:
        try:
            facts = json.loads(facts_payload)
        except json.JSONDecodeError as exc:
            raise AIResponseError(f"FACTS_JSON is not valid JSON: {exc}") from exc

        period = facts.get("period", "the analysed period")
        summary = (
            f"Over {period} the business recorded a net revenue of "
            f"{facts.get('net_revenue')} across {facts.get('order_count')} orders "
            f"(average order value {facts.get('average_order_value')})."
        )
        positives: list[str] = []
        top_products = facts.get("top_products") or []
        if top_products:
            leader = top_products[0]
            positives.append(
                f"{leader.get('product')} led net revenue with {leader.get('net_revenue')}"
            )
        trend = facts.get("monthly_trend") or []
        if len(trend) >= 2:
            first, last = trend[0], trend[-1]
            positives.append(
                f"net revenue moved from {first.get('net_revenue')} in {first.get('period')} "
                f"to {last.get('net_revenue')} in {last.get('period')}"
            )
        rate = facts.get("cancellation_rate_pct", 0.0)
        watch = (
            [f"cancelled or refunded orders accounted for {rate}% of all orders"]
            if rate
            else []
        )
        negatives: list[str] = []
        for change in facts.get("period_change") or []:
            if change.get("change_pct") is None:
                continue
            direction = "up" if change.get("change", 0) >= 0 else "down"
            negatives.append(
                f"{change.get('metric')} was {direction} {abs(change.get('change_pct'))}% "
                f"versus the previous period"
            )
        management = (
            f"protect momentum in {top_products[0].get('product')} while reviewing "
            f"{top_products[-1].get('product')} performance"
            if len(top_products) >= 2
            else "keep focusing on the strongest product lines"
        )
        return json.dumps(
            {
                "executive_summary": summary,
                "positive_trends": positives,
                "negative_trends": negatives[:5],
                "watch_items": watch,
                "management_summary": management,
            }
        )

    def _classify(self, system: str, items_payload: str) -> str:
        try:
            items = json.loads(items_payload)
        except json.JSONDecodeError as exc:
            raise AIResponseError(f"ITEMS_JSON is not valid JSON: {exc}") from exc
        categories_payload = _marker_payload(CATEGORIES_MARKER, system)
        try:
            categories = json.loads(categories_payload) if categories_payload else []
        except json.JSONDecodeError:
            categories = []
        label = str(categories[0]) if categories else "neutral"
        return json.dumps(
            [
                {"index": int(item.get("index", position)), "category": label}
                for position, item in enumerate(items)
            ]
        )


def build_provider(settings: AISettings) -> ChatProvider:
    """Instantiate the configured provider.

    Raises
    ------
    AIConfigurationError
        When the AI layer is disabled, credentials are missing, or an unknown
        provider name is configured.
    """
    if not settings.is_configured:
        raise AIConfigurationError(
            settings.unavailable_reason or "the AI layer is not configured"
        )
    if settings.provider == "mock":
        return MockProvider(settings)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise AIConfigurationError(f"unsupported AI_PROVIDER {settings.provider!r}")


__all__ = [
    "CATEGORIES_MARKER",
    "CHAT_COMPLETIONS_PATH",
    "FACTS_MARKER",
    "ITEMS_MARKER",
    "ChatProvider",
    "MockProvider",
    "OpenAICompatibleProvider",
    "build_provider",
    "parse_json_response",
    "strip_code_fences",
]