"""Google OAuth2 authentication for Colab API access.

Requires a Google OAuth2 desktop/native client configured for the
Colaboratory API scope.  Credentials are read from environment variables:

    COLAB_CLIENT_ID      – OAuth2 client ID
    COLAB_CLIENT_SECRET  – OAuth2 client secret

Flow:
1. User clicks "Connect Google" → browser opens Google auth page
2. User authenticates with Google
3. Google redirects to localhost callback
4. Exchange code for access + refresh tokens
5. Tokens stored in ~/.dialeng/colab_tokens.json
"""
import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
COLAB_SCOPES = [
    "profile",
    "email",
    "https://www.googleapis.com/auth/colaboratory"
]
TOKEN_DIR = Path.home() / ".dialeng"
TOKEN_FILE = TOKEN_DIR / "colab_tokens.json"
REFRESH_BUFFER_SECONDS = 300  # Refresh 5 minutes before expiry



@dataclass
class ColabTokens:
    """OAuth2 token set for Colab API access."""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        """Check if token needs refresh (5 min buffer)."""
        return time.time() >= self.expires_at - REFRESH_BUFFER_SECONDS

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ColabTokens':
        return cls(
            access_token=d["access_token"],
            refresh_token=d["refresh_token"],
            expires_at=d["expires_at"],
            token_type=d.get("token_type", "Bearer"),
        )


class ColabNotAuthenticatedError(Exception):
    """Raised when Colab operations are attempted without authentication."""
    pass


class ColabAuthService:
    """Manages Google OAuth2 for Colab API access.

    Requires COLAB_CLIENT_ID and COLAB_CLIENT_SECRET environment variables.
    Call get_auth_url() to start the OAuth flow, then handle_callback() to
    exchange the authorization code for tokens.
    """

    def __init__(self, redirect_uri: str = "http://localhost:8000/auth/google/callback"):
        self.client_id = os.environ.get("COLAB_CLIENT_ID", "")
        self.client_secret = os.environ.get("COLAB_CLIENT_SECRET", "")
        if not self.client_id or not self.client_secret:
            logger.warning(
                "COLAB_CLIENT_ID and COLAB_CLIENT_SECRET environment variables are required "
                "for Google Colab integration. Colab features will be unavailable."
            )
        self.redirect_uri = redirect_uri
        self._tokens: Optional[ColabTokens] = None
        self._load_tokens()

    @property
    def is_authenticated(self) -> bool:
        """Whether we have valid (or refreshable) tokens."""
        return self._tokens is not None

    def get_auth_url(self, state: str = "", redirect_uri: str = None) -> str:
        """Generate Google OAuth2 authorization URL.

        Args:
            state: CSRF protection token
            redirect_uri: Override redirect URI (e.g. from current request)

        Returns:
            Full authorization URL to redirect user to

        Raises:
            ColabNotAuthenticatedError: If OAuth credentials are not configured
        """
        if not self.client_id or not self.client_secret:
            raise ColabNotAuthenticatedError(
                "COLAB_CLIENT_ID and COLAB_CLIENT_SECRET environment variables must be set "
                "to use Google Colab integration."
            )
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri or self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(COLAB_SCOPES),
            "access_type": "offline",  # Get refresh token
            "prompt": "consent",  # Always show consent to get refresh token
        }
        if state:
            params["state"] = state
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def handle_callback(self, code: str, redirect_uri: str = None) -> ColabTokens:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from Google OAuth callback
            redirect_uri: Must match the URI used in get_auth_url()

        Returns:
            ColabTokens with access and refresh tokens

        Raises:
            httpx.HTTPStatusError: If token exchange fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": redirect_uri or self.redirect_uri,
                },
            )
            response.raise_for_status()
            data = response.json()

        expires_in = data.get("expires_in", 3600)
        self._tokens = ColabTokens(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            expires_at=time.time() + expires_in,
            token_type=data.get("token_type", "Bearer"),
        )
        self._save_tokens()
        logger.info("Colab OAuth tokens saved successfully")
        return self._tokens

    async def get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed.

        Returns:
            Valid access token string

        Raises:
            ColabNotAuthenticatedError: If not authenticated
        """
        if self._tokens is None:
            raise ColabNotAuthenticatedError("Not authenticated with Google. Click 'Connect Google' to sign in.")
        if self._tokens.is_expired:
            await self._refresh_token()
        return self._tokens.access_token

    async def _refresh_token(self):
        """Refresh the access token using the refresh token."""
        if not self._tokens or not self._tokens.refresh_token:
            raise ColabNotAuthenticatedError("No refresh token available. Please re-authenticate.")

        logger.info("Refreshing Colab OAuth access token")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._tokens.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            response.raise_for_status()
            data = response.json()

        expires_in = data.get("expires_in", 3600)
        self._tokens.access_token = data["access_token"]
        self._tokens.expires_at = time.time() + expires_in
        # Google may return a new refresh token
        if "refresh_token" in data:
            self._tokens.refresh_token = data["refresh_token"]
        self._save_tokens()
        logger.info("Colab OAuth token refreshed successfully")

    def _load_tokens(self):
        """Load tokens from disk."""
        if TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text())
                self._tokens = ColabTokens.from_dict(data)
                logger.info("Loaded Colab OAuth tokens from disk")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load Colab tokens: {e}")
                self._tokens = None

    def _save_tokens(self):
        """Save tokens to disk."""
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(self._tokens.to_dict(), indent=2))
        # Set restrictive permissions (owner read/write only)
        TOKEN_FILE.chmod(0o600)

    def logout(self):
        """Clear stored tokens."""
        self._tokens = None
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
        logger.info("Colab OAuth tokens cleared")
