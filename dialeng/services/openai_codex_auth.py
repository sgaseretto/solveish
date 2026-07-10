"""OpenAI Codex subscription OAuth and credential persistence.

This is a Dialeng-owned implementation of the PKCE flow used by Codex CLI
clients. It intentionally does not authenticate the public OpenAI Platform API.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_AUTH_CLAIM = "https://api.openai.com/auth"
AUTH_ENV_VAR = "DIALENG_OPENAI_CODEX_ACCESS_TOKEN"
LOGIN_TIMEOUT_SECONDS = 300
REFRESH_MARGIN_SECONDS = 60


@dataclass
class OpenAICodexCredentials:
    access: str
    refresh: str = ""
    expires_at: float = 0.0
    account_id: str = ""
    source: str = "stored"

    def expired(self, margin: int = REFRESH_MARGIN_SECONDS) -> bool:
        return bool(self.expires_at and time.time() >= self.expires_at - margin)


@dataclass
class PendingOAuthFlow:
    state: str
    verifier: str
    authorization_url: str
    created_at: float
    callback_available: bool = False


def _decode_jwt(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Codex access token is not a JWT")
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except Exception as exc:
        raise ValueError("Codex access token has an invalid JWT payload") from exc


def inspect_access_token(token: str) -> tuple[str, float]:
    """Return the ChatGPT account ID and expiry from a Codex OAuth token."""
    payload = _decode_jwt(token)
    auth = payload.get(JWT_AUTH_CLAIM, {})
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    if not account_id:
        raise ValueError("Token is not an OpenAI Codex subscription token")
    expires_at = float(payload.get("exp") or 0)
    return str(account_id), expires_at


def parse_authorization_input(raw: str) -> tuple[str, str]:
    """Parse a redirect URL, query string, code#state pair, or bare code."""
    value = raw.strip()
    if not value:
        return "", ""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        params = parse_qs(parsed.query)
        return params.get("code", [""])[0], params.get("state", [""])[0]
    if "#" in value:
        code, state = value.split("#", 1)
        return code, state
    if "code=" in value:
        params = parse_qs(value)
        return params.get("code", [""])[0], params.get("state", [""])[0]
    return value, ""


class OpenAICodexAuthService:
    """Server-global Codex OAuth state with private, atomic persistence."""

    def __init__(self, auth_path: Optional[Path] = None) -> None:
        self.auth_path = auth_path or Path.home() / ".dialeng" / "auth.json"
        self.pending: Optional[PendingOAuthFlow] = None
        self._refresh_lock = asyncio.Lock()
        self._callback_server: Optional[HTTPServer] = None
        self._callback_payload: Optional[tuple[str, str, str]] = None
        self._callback_lock = threading.Lock()
        self.last_error: Optional[str] = None

    def _read_auth_file(self) -> dict[str, Any]:
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_auth_file(self, data: dict[str, Any]) -> None:
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix="auth-", suffix=".json", dir=self.auth_path.parent)
        tmp_path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self.auth_path)
            os.chmod(self.auth_path, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load_credentials(self) -> Optional[OpenAICodexCredentials]:
        env_token = os.environ.get(AUTH_ENV_VAR, "").strip()
        if env_token:
            try:
                account_id, expires_at = inspect_access_token(env_token)
                return OpenAICodexCredentials(
                    access=env_token,
                    expires_at=expires_at,
                    account_id=account_id,
                    source=f"env:{AUTH_ENV_VAR}",
                )
            except ValueError as exc:
                self.last_error = str(exc)
                logger.warning("Ignoring invalid %s: %s", AUTH_ENV_VAR, exc)

        raw = self._read_auth_file().get("openai_codex")
        if not isinstance(raw, dict):
            return None
        try:
            credentials = OpenAICodexCredentials(
                access=str(raw["access"]),
                refresh=str(raw.get("refresh", "")),
                expires_at=float(raw.get("expires_at", 0)),
                account_id=str(raw.get("account_id", "")),
                source="stored",
            )
            account_id, jwt_expiry = inspect_access_token(credentials.access)
            if credentials.account_id and credentials.account_id != account_id:
                raise ValueError("stored account ID does not match the access token")
            credentials.account_id = credentials.account_id or account_id
            credentials.expires_at = credentials.expires_at or jwt_expiry
            return credentials
        except (KeyError, TypeError, ValueError) as exc:
            self.last_error = f"Stored Codex credentials are invalid: {exc}"
            return None

    def save_credentials(self, credentials: OpenAICodexCredentials) -> None:
        data = self._read_auth_file()
        stored = asdict(credentials)
        stored.pop("source", None)
        data["openai_codex"] = stored
        self._write_auth_file(data)

    def logout(self) -> bool:
        data = self._read_auth_file()
        removed = data.pop("openai_codex", None) is not None
        if data:
            self._write_auth_file(data)
        else:
            self.auth_path.unlink(missing_ok=True)
        self.pending = None
        self.last_error = None
        self._stop_callback_server()
        return removed

    @property
    def is_authenticated(self) -> bool:
        credentials = self.load_credentials()
        return bool(credentials and (not credentials.expired() or credentials.refresh))

    def status(self) -> dict[str, Any]:
        credentials = self.load_credentials()
        pending = self.pending
        if pending and time.time() - pending.created_at > LOGIN_TIMEOUT_SECONDS:
            self.pending = None
            pending = None
            self._stop_callback_server()
        return {
            "provider": "openai_codex",
            "authenticated": bool(credentials and (not credentials.expired() or credentials.refresh)),
            "source": credentials.source if credentials else "none",
            "account_id": credentials.account_id if credentials else None,
            "expires_at": credentials.expires_at if credentials else None,
            "pending": pending is not None,
            "callback_available": bool(pending and pending.callback_available),
            "authorization_url": pending.authorization_url if pending else None,
            "error": self.last_error,
        }

    def start_login(self) -> dict[str, Any]:
        if self.pending and time.time() - self.pending.created_at <= LOGIN_TIMEOUT_SECONDS:
            return self.status()
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).decode("ascii").rstrip("=")
        state = secrets.token_hex(24)
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "dialeng",
        }
        pending = PendingOAuthFlow(
            state=state,
            verifier=verifier,
            authorization_url=f"{AUTHORIZE_URL}?{urlencode(params)}",
            created_at=time.time(),
        )
        self.pending = pending
        self.last_error = None
        pending.callback_available = self._start_callback_server()
        return self.status()

    def _start_callback_server(self) -> bool:
        service = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                code = params.get("code", [""])[0]
                state = params.get("state", [""])[0]
                error = params.get("error", [""])[0]
                with service._callback_lock:
                    service._callback_payload = (code, state, error)
                ok = bool(code and service.pending and state == service.pending.state and not error)
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                message = "Authentication received. Return to Dialeng." if ok else "Authentication failed. Return to Dialeng and retry."
                self.wfile.write(f"<!doctype html><title>Dialeng OAuth</title><p>{message}</p>".encode())

            def log_message(self, _format: str, *args: Any) -> None:
                return

        try:
            self._stop_callback_server()
            server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
            server.timeout = LOGIN_TIMEOUT_SECONDS
            self._callback_server = server

            def run_once() -> None:
                try:
                    server.handle_request()
                finally:
                    server.server_close()
                    if service._callback_server is server:
                        service._callback_server = None

            threading.Thread(target=run_once, name="dialeng-codex-oauth", daemon=True).start()
            return True
        except OSError as exc:
            logger.info("Codex OAuth callback port unavailable; manual completion remains available: %s", exc)
            self._callback_server = None
            return False

    def _stop_callback_server(self) -> None:
        server = self._callback_server
        self._callback_server = None
        if server:
            try:
                server.server_close()
            except OSError:
                pass

    async def complete_loopback_if_ready(self) -> bool:
        with self._callback_lock:
            payload = self._callback_payload
            self._callback_payload = None
        if not payload:
            return False
        code, state, error = payload
        if error:
            self.last_error = f"OpenAI authorization failed: {error}"
            self.pending = None
            return False
        await self.complete_authorization(code, state)
        return True

    async def complete_authorization(self, raw_code: str, state: str = "") -> OpenAICodexCredentials:
        code, parsed_state = parse_authorization_input(raw_code)
        state = state or parsed_state
        pending = self.pending
        if not pending or time.time() - pending.created_at > LOGIN_TIMEOUT_SECONDS:
            raise ValueError("No active OpenAI login flow or the flow expired")
        if not state or state != pending.state:
            raise ValueError("OAuth state mismatch")
        if not code:
            raise ValueError("Missing authorization code")
        credentials = await self._exchange_code(code, pending.verifier)
        self.save_credentials(credentials)
        self.pending = None
        self.last_error = None
        self._stop_callback_server()
        return credentials

    async def _exchange_code(self, code: str, verifier: str) -> OpenAICodexCredentials:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(TOKEN_URL, data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI token exchange failed ({response.status_code})")
        return self._credentials_from_token_response(response.json())

    @staticmethod
    def _credentials_from_token_response(
        data: dict[str, Any], fallback_refresh: str = ""
    ) -> OpenAICodexCredentials:
        access = str(data.get("access_token", ""))
        refresh = str(data.get("refresh_token") or fallback_refresh)
        expires_in = data.get("expires_in")
        if not access or not refresh or not isinstance(expires_in, (int, float)):
            raise RuntimeError("OpenAI token response was incomplete")
        account_id, jwt_expiry = inspect_access_token(access)
        return OpenAICodexCredentials(
            access=access,
            refresh=refresh,
            expires_at=time.time() + float(expires_in) if expires_in else jwt_expiry,
            account_id=account_id,
            source="stored",
        )

    async def get_valid_credentials(self, force_refresh: bool = False) -> Optional[OpenAICodexCredentials]:
        credentials = self.load_credentials()
        if not credentials:
            return None
        if credentials.source.startswith("env:"):
            return None if credentials.expired() else credentials
        if not force_refresh and not credentials.expired():
            return credentials
        if not credentials.refresh:
            return None
        async with self._refresh_lock:
            latest = self.load_credentials()
            if latest and not force_refresh and not latest.expired():
                return latest
            refresh_token = (latest or credentials).refresh
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(TOKEN_URL, data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CLIENT_ID,
                }, headers={"Content-Type": "application/x-www-form-urlencoded"})
            if response.status_code != 200:
                self.last_error = f"OpenAI token refresh failed ({response.status_code})"
                return None
            refreshed = self._credentials_from_token_response(
                response.json(), fallback_refresh=refresh_token
            )
            self.save_credentials(refreshed)
            self.last_error = None
            return refreshed


_auth_service: Optional[OpenAICodexAuthService] = None


def get_openai_codex_auth_service() -> OpenAICodexAuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = OpenAICodexAuthService()
    return _auth_service


def reset_openai_codex_auth_service() -> None:
    global _auth_service
    _auth_service = None
