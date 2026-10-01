"""Google API authentication.

Two modes are supported, both configured through environment variables:

``service_account``
    A service-account key file (or the key JSON itself via
    ``GS_SERVICE_ACCOUNT_JSON``). This is the recommended mode for automation:
    no browser, no interactive consent, and the key can be stored as a CI secret.
    The target spreadsheet has to be shared with the service-account e-mail.

``oauth``
    An installed-application OAuth flow. The client-secrets file is used once to
    obtain a user token which is then cached on disk and refreshed automatically.

Only the *Sheets* scope is requested: the application never needs Drive access,
so it does not ask for it (least privilege).

Credentials (files, tokens, keys) are never logged, printed or copied anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from src.config.logging_config import get_logger
from src.config.settings import GoogleAuthSettings, GoogleSheetsSettings
from src.utils.errors import AuthenticationError

SHEETS_API_SERVICE_NAME = "sheets"
SHEETS_API_VERSION = "v4"


def build_credentials(auth: GoogleAuthSettings) -> Credentials:
    """Create Google credentials for the configured mode.

    Raises
    ------
    AuthenticationError
        When the configuration points at a missing/invalid credential source.
    """
    logger = get_logger("google.auth")
    if auth.mode == "service_account":
        credentials = _service_account_credentials(auth)
    else:
        credentials = _oauth_credentials(auth)
    logger.info("google credentials ready (%s)", auth.location_summary())
    return credentials


def _service_account_credentials(auth: GoogleAuthSettings) -> Credentials:
    if auth.service_account_json:
        try:
            info = json.loads(auth.service_account_json)
        except json.JSONDecodeError as exc:
            raise AuthenticationError(
                "GS_SERVICE_ACCOUNT_JSON does not contain valid JSON"
            ) from exc
        try:
            return service_account.Credentials.from_service_account_info(
                info, scopes=list(auth.scopes)
            )
        except (ValueError, KeyError) as exc:
            raise AuthenticationError(
                "GS_SERVICE_ACCOUNT_JSON is not a usable service-account key"
            ) from exc

    key_file: Path = auth.service_account_file
    if not key_file.is_file():
        raise AuthenticationError(
            "service-account key file not found: "
            f"{key_file}. Set GS_SERVICE_ACCOUNT_FILE, provide GS_SERVICE_ACCOUNT_JSON, "
            "or switch GS_AUTH_MODE to 'oauth'."
        )
    try:
        return service_account.Credentials.from_service_account_file(
            str(key_file), scopes=list(auth.scopes)
        )
    except (ValueError, OSError) as exc:
        raise AuthenticationError(
            f"service-account key file could not be loaded: {key_file}"
        ) from exc


def _oauth_credentials(auth: GoogleAuthSettings) -> Credentials:
    token_file: Path = auth.oauth_token_file
    if token_file.is_file():
        try:
            credentials = Credentials.from_authorized_user_file(str(token_file), list(auth.scopes))
        except (ValueError, OSError) as exc:
            raise AuthenticationError(
                f"cached OAuth token could not be read: {token_file}"
            ) from exc
        if credentials.valid:
            return credentials
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            _store_token(token_file, credentials)
            return credentials

    secrets_file: Path = auth.oauth_client_secrets_file
    if not secrets_file.is_file():
        raise AuthenticationError(
            "OAuth client secrets file not found: "
            f"{secrets_file}. Set GS_OAUTH_CLIENT_SECRETS_FILE or switch "
            "GS_AUTH_MODE to 'service_account'."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), list(auth.scopes))
    # Opens the system browser once; the resulting token is cached afterwards.
    credentials = flow.run_local_server(port=0)
    _store_token(token_file, credentials)
    return credentials


def _store_token(token_file: Path, credentials: Credentials) -> None:
    """Persist the refreshed/created OAuth token (never logged)."""
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    get_logger("google.auth").info("OAuth token cached at %s", token_file)


def build_sheets_service(
    settings: GoogleSheetsSettings,
    credentials: Credentials | None = None,
    *,
    discovery_cache: bool = False,
) -> Any:
    """Build a Google Sheets API v4 service object.

    ``credentials`` can be injected for testing; otherwise they are derived from
    the configured auth mode.
    """
    from googleapiclient.discovery import build

    resolved = credentials or build_credentials(settings.auth)
    try:
        return build(
            SHEETS_API_SERVICE_NAME,
            SHEETS_API_VERSION,
            credentials=resolved,
            cache_discovery=discovery_cache,
        )
    except Exception as exc:  # pragma: no cover - network/discovery failure
        raise AuthenticationError(f"could not initialise the Google Sheets service: {exc}") from exc
