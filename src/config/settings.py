"""Central configuration for the platform.

All environment dependent values live here so no other module needs to read
``os.environ`` or hard-code magic values. Use :func:`load_settings` once at the
entry point (CLI, scheduler, tests) and pass the resulting ``Settings`` object
down through the application.

Secrets are never logged: :meth:`Settings.describe` only reports whether a value
is present, never the value itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv

from src.utils.errors import ConfigurationError

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

AuthMode = Literal["service_account", "oauth"]
ScheduleMode = Literal["daily", "weekly", "interval"]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Small typed environment readers
# ---------------------------------------------------------------------------
def _raw(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _text(env: Mapping[str, str], key: str, default: str) -> str:
    return _raw(env, key) or default


def _optional_text(env: Mapping[str, str], key: str) -> str | None:
    return _raw(env, key)


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = _raw(env, key)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{key} must be a boolean (got {value!r})")


def _int(env: Mapping[str, str], key: str, default: int, *, minimum: int | None = None) -> int:
    value = _raw(env, key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer (got {value!r})") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum} (got {parsed})")
    return parsed


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _raw(env, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a number (got {value!r})") from exc


def _path(env: Mapping[str, str], key: str, default: str) -> Path:
    candidate = Path(_text(env, key, default)).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


_AI_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"

DEFAULT_AI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_AI_MODEL = "gpt-4o-mini"
# Output budget per request. The interpretation task only verbalises numbers the
# KPI engine already computed, but reasoning models spend part of this budget on
# hidden thinking before writing the answer, so the default is generous enough
# for a long narrative plus a full reasoning pass.
DEFAULT_AI_MAX_OUTPUT_TOKENS = 3000

# Reasoning control for OpenRouter-style models. The deterministic KPI engine is
# the source of truth here, so the interpretation task wants as little hidden
# reasoning as possible: "minimal" is the lowest effort that every endpoint
# accepts (OpenRouter maps unsupported levels to the nearest supported one).
# "none" fully disables thinking but is rejected by endpoints where reasoning is
# mandatory, and "auto" omits the parameter entirely (model/provider default).
DEFAULT_AI_REASONING_EFFORT = "minimal"

_REASONING_EFFORT_ALIASES: dict[str, str | None] = {
    "off": "none",
    "disable": "none",
    "disabled": "none",
    "false": "none",
    "auto": None,
    "default": None,
    "model_default": None,
}
_REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh")


def is_openrouter_host(base_url: str) -> bool:
    """True when the base URL points at OpenRouter (which accepts ``reasoning``)."""
    return (urlsplit(base_url).hostname or "").lower().endswith("openrouter.ai")


def normalize_base_url(value: str, *, context: str = "AI_BASE_URL") -> str:
    """Normalise an OpenAI-compatible API base URL.

    Accepts ``https://openrouter.ai/api/v1`` (with or without a trailing slash)
    and also tolerates a full chat-completions URL such as
    ``https://openrouter.ai/api/v1/chat/completions`` by trimming the endpoint
    path, so the provider can never build a doubled
    ``.../v1/chat/completions/chat/completions`` target.

    Raises
    ------
    ConfigurationError
        When the value is present but is not an absolute http(s) URL.
    """
    candidate = value.strip().rstrip("/")
    if candidate.lower().endswith(_AI_CHAT_COMPLETIONS_SUFFIX):
        candidate = candidate[: -len(_AI_CHAT_COMPLETIONS_SUFFIX)].rstrip("/")
    if not candidate:
        return ""
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ConfigurationError(
            f"{context} must be an absolute http(s) URL "
            f"(for example 'https://openrouter.ai/api/v1'), got {candidate!r}"
        )
    return candidate


def _optional_bool(env: Mapping[str, str], key: str) -> bool | None:
    """Tri-state boolean: None when the variable is absent (used for "auto")."""
    return None if _raw(env, key) is None else _bool(env, key, False)


def _reasoning_effort(env: Mapping[str, str]) -> str | None:
    """Parse ``AI_REASONING_EFFORT`` into an effort level or None (send nothing).

    ``auto``/``default`` mean "let the provider decide" and are turned into None;
    ``off``/``disabled``/``false`` are aliases for ``none``.
    """
    value = _text(env, "AI_REASONING_EFFORT", DEFAULT_AI_REASONING_EFFORT).lower()
    if value in _REASONING_EFFORT_ALIASES:
        return _REASONING_EFFORT_ALIASES[value]
    if value in _REASONING_EFFORTS:
        return value
    raise ConfigurationError(
        f"AI_REASONING_EFFORT must be one of {list(_REASONING_EFFORTS)} or 'auto' "
        f"(got {value!r})"
    )


def _time_of_day(env: Mapping[str, str], key: str, default: str) -> time:
    value = _text(env, key, default)
    try:
        hour_text, _, minute_text = value.partition(":")
        return time(hour=int(hour_text), minute=int(minute_text or "0"))
    except ValueError as exc:
        raise ConfigurationError(f"{key} must use HH:MM (got {value!r})") from exc


# ---------------------------------------------------------------------------
# Configuration groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoogleAuthSettings:
    """How the application authenticates against Google APIs."""

    mode: AuthMode
    service_account_file: Path
    service_account_json: str | None
    oauth_client_secrets_file: Path
    oauth_token_file: Path
    scopes: tuple[str, ...]

    def location_summary(self) -> str:
        """Human readable description that never contains secret material."""
        if self.mode == "service_account":
            if self.service_account_json:
                return "service_account key provided via environment variable"
            return f"service_account key file: {self.service_account_file}"
        return f"oauth client secrets: {self.oauth_client_secrets_file} (token cache: {self.oauth_token_file})"


@dataclass(frozen=True)
class WorksheetNames:
    """Configurable worksheet (tab) names for the output workbook."""

    raw_data: str
    clean_data: str
    data_quality: str
    kpi_summary: str
    product_analysis: str
    category_analysis: str
    regional_analysis: str
    ai_insights: str
    run_log: str

    def ordered(self) -> tuple[str, ...]:
        """Worksheet creation order (as presented to a business user)."""
        return (
            self.raw_data,
            self.clean_data,
            self.data_quality,
            self.kpi_summary,
            self.product_analysis,
            self.category_analysis,
            self.regional_analysis,
            self.ai_insights,
            self.run_log,
        )


@dataclass(frozen=True)
class GoogleSheetsSettings:
    """Spreadsheet access configuration."""

    spreadsheet_id: str | None
    raw_data_range: str
    run_log_max_rows: int
    worksheets: WorksheetNames
    auth: GoogleAuthSettings

    @property
    def is_configured(self) -> bool:
        """True when a spreadsheet id is present (otherwise: local dry-run)."""
        return bool(self.spreadsheet_id)


@dataclass(frozen=True)
class PipelineSettings:
    """Business-rule knobs for the data pipeline."""

    top_n_products: int
    revenue_outlier_threshold: float
    local_output_dir: Path

    def __post_init__(self) -> None:
        if self.top_n_products < 1:
            raise ConfigurationError("top_n_products must be >= 1")
        if self.revenue_outlier_threshold <= 0:
            raise ConfigurationError("revenue_outlier_threshold must be > 0")


@dataclass(frozen=True)
class LoggingSettings:
    """Logging destination and verbosity."""

    level: str
    log_file: Path | None

    def __post_init__(self) -> None:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in allowed:
            raise ConfigurationError(
                f"LOG_LEVEL must be one of {sorted(allowed)} (got {self.level!r})"
            )


@dataclass(frozen=True)
class AISettings:
    """AI interpretation layer configuration.

    The AI provider is fully configurable and optional. ``api_key`` is kept in
    memory only and is redacted from every log/description produced here.
    """

    enabled: bool
    provider: str
    base_url: str
    model: str
    api_key_env_var: str
    api_key: str | None
    timeout_seconds: float
    max_input_chars: int
    max_output_tokens: int
    temperature: float
    classification_enabled: bool
    classification_categories: tuple[str, ...]
    classification_text_column: str
    classification_batch_size: int
    classification_max_rows: int
    # Optional OpenRouter attribution headers (only sent when configured).
    http_referer: str | None = None
    app_title: str | None = None
    # Reasoning control (OpenRouter-style thinking models). ``None`` for
    # ``reasoning_effort`` means "send no reasoning parameter at all".
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    send_reasoning_params: bool = True

    @property
    def reasoning_payload(self) -> dict[str, object] | None:
        """Request field controlling thinking tokens (``None`` = omit it).

        The KPI engine already computed every authoritative number, so hidden
        reasoning is wasted budget for this task; the default configuration
        therefore asks for ``effort="none"``. The field is only sent when
        :attr:`send_reasoning_params` is true (automatic for OpenRouter hosts)
        so gateways whose models do not understand ``reasoning`` are unaffected.
        Reasoning traces are never consumed by this platform, so they are
        excluded from the response.
        """
        if not self.send_reasoning_params or self.reasoning_effort is None:
            return None
        payload: dict[str, object] = {"effort": self.reasoning_effort, "exclude": True}
        if self.reasoning_max_tokens is not None:
            payload["max_tokens"] = self.reasoning_max_tokens
        return payload

    @property
    def provider_label(self) -> str:
        """Stable label used in spreadsheets and logs (never contains secrets)."""
        return f"{self.provider}:{self.model}"

    @property
    def is_configured(self) -> bool:
        """True when the AI layer can actually be called.

        The ``mock`` provider is intentionally considered configured: it is a
        deterministic, offline provider used for demos and tests.
        """
        if not self.enabled:
            return False
        if self.provider == "mock":
            return True
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str | None:
        """Explanation for the pipeline when AI cannot run (None when usable)."""
        if not self.enabled:
            return "disabled via AI_ENABLED=false"
        if self.provider != "mock" and not self.api_key:
            return f"credentials not configured (environment variable {self.api_key_env_var} is empty)"
        return None


@dataclass(frozen=True)
class SchedulerSettings:
    """APScheduler configuration for recurring pipeline runs."""

    mode: ScheduleMode
    time_of_day: time
    day_of_week: int
    interval_minutes: int
    timezone: str

    def __post_init__(self) -> None:
        if not 0 <= self.day_of_week <= 6:
            raise ConfigurationError("SCHEDULE_DAY_OF_WEEK must be between 0 (Monday) and 6 (Sunday)")
        if self.mode == "interval" and self.interval_minutes < 1:
            raise ConfigurationError("SCHEDULE_INTERVAL_MINUTES must be >= 1")

    def describe(self) -> str:
        """Human readable schedule description used in logs and the Run_Log."""
        if self.mode == "daily":
            return f"daily at {self.time_of_day.strftime('%H:%M')} ({self.timezone})"
        if self.mode == "weekly":
            days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
            return (
                f"weekly on {days[self.day_of_week]} at "
                f"{self.time_of_day.strftime('%H:%M')} ({self.timezone})"
            )
        return f"every {self.interval_minutes} minutes ({self.timezone})"


@dataclass(frozen=True)
class Settings:
    """Aggregate application configuration."""

    project_root: Path
    google_sheets: GoogleSheetsSettings
    pipeline: PipelineSettings
    logging: LoggingSettings
    ai: AISettings
    scheduler: SchedulerSettings

    def describe(self) -> dict[str, object]:
        """Redacted configuration summary suitable for logging."""
        return {
            "project_root": str(self.project_root),
            "spreadsheet_id_configured": self.google_sheets.is_configured,
            "auth_mode": self.google_sheets.auth.mode,
            "auth_location": self.google_sheets.auth.location_summary(),
            "raw_data_range": self.google_sheets.raw_data_range,
            "log_level": self.logging.level,
            "log_file": str(self.logging.log_file) if self.logging.log_file else None,
            "local_output_dir": str(self.pipeline.local_output_dir),
            "top_n_products": self.pipeline.top_n_products,
            "ai_enabled": self.ai.enabled,
            "ai_provider": self.ai.provider,
            "ai_model": self.ai.model,
            "ai_base_url": self.ai.base_url,
            "ai_configured": self.ai.is_configured,
            "ai_api_key_present": bool(self.ai.api_key),
            "schedule": self.scheduler.describe(),
        }


def _build_auth_settings(env: Mapping[str, str]) -> GoogleAuthSettings:
    mode = _text(env, "GS_AUTH_MODE", "service_account").lower()
    if mode not in ("service_account", "oauth"):
        raise ConfigurationError("GS_AUTH_MODE must be 'service_account' or 'oauth'")
    scopes = tuple(
        scope
        for scope in _text(env, "GS_SCOPES", "https://www.googleapis.com/auth/spreadsheets").split()
        if scope
    )
    return GoogleAuthSettings(
        mode=mode,  # type: ignore[arg-type]
        service_account_file=_path(env, "GS_SERVICE_ACCOUNT_FILE", "credentials/service_account.json"),
        service_account_json=_optional_text(env, "GS_SERVICE_ACCOUNT_JSON"),
        oauth_client_secrets_file=_path(
            env, "GS_OAUTH_CLIENT_SECRETS_FILE", "credentials/client_secret.json"
        ),
        oauth_token_file=_path(env, "GS_OAUTH_TOKEN_FILE", "credentials/token.json"),
        scopes=scopes,
    )


def _build_worksheet_names(env: Mapping[str, str]) -> WorksheetNames:
    return WorksheetNames(
        raw_data=_text(env, "GS_SHEET_RAW_DATA", "Raw_Data"),
        clean_data=_text(env, "GS_SHEET_CLEAN_DATA", "Clean_Data"),
        data_quality=_text(env, "GS_SHEET_DATA_QUALITY", "Data_Quality"),
        kpi_summary=_text(env, "GS_SHEET_KPI_SUMMARY", "KPI_Summary"),
        product_analysis=_text(env, "GS_SHEET_PRODUCT_ANALYSIS", "Product_Analysis"),
        category_analysis=_text(env, "GS_SHEET_CATEGORY_ANALYSIS", "Category_Analysis"),
        regional_analysis=_text(env, "GS_SHEET_REGIONAL_ANALYSIS", "Regional_Analysis"),
        ai_insights=_text(env, "GS_SHEET_AI_INSIGHTS", "AI_Insights"),
        run_log=_text(env, "GS_SHEET_RUN_LOG", "Run_Log"),
    )


def _validate_openrouter_model(base_url: str, model: str) -> None:
    """Catch the most common OpenRouter misconfiguration early.

    OpenRouter addresses models as ``<vendor>/<model>[:variant]`` (for example
    ``qwen/qwen3.8-27b:free``). A slug without the vendor prefix is rejected by
    the API, which surfaced as a confusing provider error at runtime, so it is
    reported here as a clear configuration problem instead.
    """
    host = (urlsplit(base_url).hostname or "").lower()
    if not host.endswith("openrouter.ai"):
        return
    if not model:
        raise ConfigurationError("AI_MODEL must not be empty when AI_PROVIDER=openai_compatible")
    if "/" not in model:
        raise ConfigurationError(
            "AI_MODEL must use the OpenRouter '<vendor>/<model>[:variant]' form "
            f"(for example 'qwen/qwen3.8-27b:free'); got {model!r}"
        )


def _build_ai_settings(env: Mapping[str, str]) -> AISettings:
    categories = tuple(
        category.strip()
        for category in _text(
            env,
            "AI_CLASSIFICATION_CATEGORIES",
            "positive,negative,neutral,complaint,product_feedback,delivery_feedback",
        ).split(",")
        if category.strip()
    )
    api_key_env_var = _text(env, "AI_API_KEY_ENV", "OPENAI_API_KEY")
    provider = _text(env, "AI_PROVIDER", "openai_compatible").lower()
    base_url = normalize_base_url(_text(env, "AI_BASE_URL", DEFAULT_AI_BASE_URL))
    model = _text(env, "AI_MODEL", DEFAULT_AI_MODEL)
    if provider == "openai_compatible":
        _validate_openrouter_model(base_url, model)
    # OpenRouter accepts the ``reasoning`` field; other gateways may reject
    # unknown parameters, so reasoning is only sent automatically for OpenRouter
    # hosts unless AI_SEND_REASONING_PARAMS says otherwise.
    send_reasoning = _optional_bool(env, "AI_SEND_REASONING_PARAMS")
    if send_reasoning is None:
        send_reasoning = provider == "openai_compatible" and is_openrouter_host(base_url)
    return AISettings(
        enabled=_bool(env, "AI_ENABLED", True),
        provider=provider,
        base_url=base_url,
        model=model,
        api_key_env_var=api_key_env_var,
        api_key=_optional_text(env, api_key_env_var),
        timeout_seconds=_float(env, "AI_TIMEOUT_SECONDS", 60.0),
        max_input_chars=_int(env, "AI_MAX_INPUT_CHARS", 12000, minimum=500),
        max_output_tokens=_int(
            env, "AI_MAX_OUTPUT_TOKENS", DEFAULT_AI_MAX_OUTPUT_TOKENS, minimum=64
        ),
        temperature=_float(env, "AI_TEMPERATURE", 0.2),
        classification_enabled=_bool(env, "AI_CLASSIFICATION_ENABLED", True),
        classification_categories=categories,
        classification_text_column=_text(env, "AI_CLASSIFICATION_TEXT_COLUMN", "feedback"),
        classification_batch_size=_int(env, "AI_CLASSIFICATION_BATCH_SIZE", 20, minimum=1),
        classification_max_rows=_int(env, "AI_CLASSIFICATION_MAX_ROWS", 200, minimum=1),
        http_referer=_optional_text(env, "AI_HTTP_REFERER"),
        app_title=_optional_text(env, "AI_APP_TITLE"),
        reasoning_effort=_reasoning_effort(env),
        reasoning_max_tokens=(
            None
            if _raw(env, "AI_REASONING_MAX_TOKENS") is None
            else _int(env, "AI_REASONING_MAX_TOKENS", 0, minimum=1)
        ),
        send_reasoning_params=send_reasoning,
    )


def _build_scheduler_settings(env: Mapping[str, str]) -> SchedulerSettings:
    mode = _text(env, "SCHEDULE_MODE", "daily").lower()
    if mode not in ("daily", "weekly", "interval"):
        raise ConfigurationError("SCHEDULE_MODE must be 'daily', 'weekly' or 'interval'")
    return SchedulerSettings(
        mode=mode,  # type: ignore[arg-type]
        time_of_day=_time_of_day(env, "SCHEDULE_TIME", "07:30"),
        day_of_week=_int(env, "SCHEDULE_DAY_OF_WEEK", 0),
        interval_minutes=_int(env, "SCHEDULE_INTERVAL_MINUTES", 60),
        timezone=_text(env, "SCHEDULE_TIMEZONE", "UTC"),
    )


def load_settings(
    env_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Build a :class:`Settings` object.

    Parameters
    ----------
    env_file:
        Optional path to a ``.env`` file. Defaults to ``<project_root>/.env``.
        The file is optional - real environment variables always win.
    env:
        Explicit environment mapping. Tests use this to avoid touching the real
        process environment; when provided, ``env_file`` loading is skipped and
        ``env`` is used as-is (missing keys fall back to defaults).

    Raises
    ------
    ConfigurationError
        When a value is present but malformed.
    """
    if env is not None:
        source: Mapping[str, str] = env
    else:
        dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
        source = os.environ

    log_file_value = _optional_text(source, "LOG_FILE")
    logging_settings = LoggingSettings(
        level=_text(source, "LOG_LEVEL", "INFO").upper(),
        log_file=(
            None
            if log_file_value is None
            else _path(source, "LOG_FILE", "logs/pipeline.log")
        ),
    )

    ai_settings = _build_ai_settings(source)
    if ai_settings.enabled and ai_settings.provider not in ("openai_compatible", "mock"):
        raise ConfigurationError(
            "AI_PROVIDER must be 'openai_compatible' or 'mock' "
            f"(got {ai_settings.provider!r})"
        )
    if ai_settings.provider == "openai_compatible" and not ai_settings.base_url:
        raise ConfigurationError(
            "AI_BASE_URL must not be empty for the openai_compatible provider "
            "(for example 'https://openrouter.ai/api/v1')"
        )

    return Settings(
        project_root=PROJECT_ROOT,
        google_sheets=GoogleSheetsSettings(
            spreadsheet_id=_optional_text(source, "GS_SPREADSHEET_ID"),
            raw_data_range=_text(source, "GS_RAW_DATA_RANGE", "A1:Z10000"),
            run_log_max_rows=_int(source, "GS_RUN_LOG_MAX_ROWS", 500, minimum=10),
            worksheets=_build_worksheet_names(source),
            auth=_build_auth_settings(source),
        ),
        pipeline=PipelineSettings(
            top_n_products=_int(source, "TOP_N_PRODUCTS", 5, minimum=1),
            revenue_outlier_threshold=_float(source, "REVENUE_OUTLIER_THRESHOLD", 10000.0),
            local_output_dir=_path(source, "LOCAL_OUTPUT_DIR", "output"),
        ),
        logging=logging_settings,
        ai=ai_settings,
        scheduler=_build_scheduler_settings(source),
    )
