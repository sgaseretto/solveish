"""Colab REST API client.

Handles authenticated requests to the Colab API including:
- Two-step XSRF pattern (GET token, POST with token) for assign/unassign
- Two backend domains: ColabApiDomain (tunnel mgmt) and ColabGapiDomain (user info, tokens)
- XSSI protection prefix stripping
- Runtime assignment, unassignment, keep-alive, session management

API reference reverse-engineered from:
- https://github.com/googlecolab/colab-vscode (Apache 2.0, Google LLC)
- https://github.com/xeodou/colab-cli
"""
import json
import logging
import uuid
from typing import Optional, List
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

COLAB_API_BASE = "https://colab.research.google.com"   # Tunnel management
COLAB_GAPI_BASE = "https://colab.pa.googleapis.com"    # User info, tokens, assignments

# Colab APIs prefix JSON with this to prevent XSSI attacks
XSSI_PREFIX = ")]}'\n"


def _strip_xssi(response_text: str) -> str:
    """Strip XSSI protection prefix from Colab API responses."""
    if response_text.startswith(XSSI_PREFIX):
        return response_text[len(XSSI_PREFIX):]
    if response_text.startswith(")]}'"):
        return response_text[4:]
    return response_text


def _make_notebook_hash() -> str:
    """Generate a notebook hash (web-safe base64 UUID) for runtime assignment."""
    raw = uuid.uuid4().hex
    # Convert to web-safe base64-like format: replace hyphens with underscores, pad to 44 chars
    formatted = f"{raw[:8]}_{raw[8:12]}_{raw[12:16]}_{raw[16:20]}_{raw[20:]}"
    return formatted.ljust(44, '.')


@dataclass
class RuntimeProxyInfo:
    """Proxy connection info for a Colab runtime."""
    token: str                  # Proxy token for WebSocket/API auth
    url: str                    # Runtime proxy URL (e.g. https://xxx-colab.googleusercontent.com/)
    token_expires_seconds: int = 3600  # Token TTL


@dataclass
class ColabAssignment:
    """Result of a kernel assignment from the Colab API."""
    endpoint: str              # Runtime endpoint ID (used in keep-alive, unassign, etc.)
    proxy_info: RuntimeProxyInfo
    accelerator: str = ""      # e.g. "T4", "A100"
    outcome: int = 0           # 0=undefined, 4=success, 1=quota_denied, 2=quota_exceeded
    idle_timeout: int = 1800   # Frontend idle timeout in seconds


@dataclass
class JupyterSession:
    """A Jupyter notebook session on a Colab runtime."""
    session_id: str
    kernel_id: str
    name: str = "colab"


class ColabAPIError(Exception):
    """Error from Colab API."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class ColabTooManyAssignmentsError(ColabAPIError):
    """User has too many active runtimes."""
    pass


class ColabInsufficientQuotaError(ColabAPIError):
    """User has exceeded their compute quota."""
    pass


class ColabAPIClient:
    """Client for Colab's REST API.

    Uses two backend domains:
    - COLAB_API_BASE (colab.research.google.com): tunnel management, XSRF-protected endpoints
    - COLAB_GAPI_BASE (colab.pa.googleapis.com): user info, proxy tokens, assignment listing
    """

    def __init__(self, auth_service):
        self.auth = auth_service

    def _base_headers(self, token: str) -> dict:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Colab-Client-Agent": "dialeng",
        }

    async def _api_request(self, method: str, path: str, extra_headers: dict = None,
                           **kwargs) -> httpx.Response:
        """Make authenticated request to ColabApiDomain (auto-appends ?authuser=0)."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)
        if extra_headers:
            headers.update(extra_headers)

        url = f"{COLAB_API_BASE}{path}"
        # Append authuser=0
        sep = "&" if "?" in url else "?"
        url += f"{sep}authuser=0"

        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    async def _gapi_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make authenticated request to ColabGapiDomain."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)

        url = f"{COLAB_GAPI_BASE}{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    async def _runtime_request(self, method: str, proxy_url: str, path: str,
                                proxy_token: str, **kwargs) -> httpx.Response:
        """Make request to a runtime proxy (Jupyter API on the runtime)."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)
        headers["X-Colab-Runtime-Proxy-Token"] = proxy_token
        headers["Content-Type"] = "application/json"

        url = f"{proxy_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    # ==================== Assignment ====================

    async def assign_kernel(self, variant: str = "", accelerator: str = "",
                            shape: str = "") -> ColabAssignment:
        """Request a new Colab runtime using two-step XSRF pattern.

        Args:
            variant: "GPU", "TPU", or "" for default CPU
            accelerator: Specific accelerator (e.g. "T4", "A100")
            shape: "hm" for high-memory, "" for standard

        Returns:
            ColabAssignment with endpoint and proxy info
        """
        nbh = _make_notebook_hash()

        # Build query params
        params = f"?nbh={nbh}"
        if variant:
            params += f"&variant={variant}"
        if accelerator:
            params += f"&accelerator={accelerator}"
        if shape:
            params += f"&shape={shape}"

        # Step 1: GET to obtain XSRF token (or existing assignment)
        resp = await self._api_request("GET", f"/tun/m/assign{params}")
        self._check_response(resp)
        data = json.loads(_strip_xssi(resp.text))

        # Check if already assigned (no 'token' field = already have assignment)
        if "runtimeProxyInfo" in data:
            logger.info("Reusing existing Colab runtime assignment")
            return self._parse_assignment(data)

        xsrf_token = data.get("token", "")
        if not xsrf_token:
            raise ColabAPIError("No XSRF token in assign GET response")

        # Step 2: POST with XSRF token to create assignment
        resp = await self._api_request(
            "POST", f"/tun/m/assign{params}",
            extra_headers={"X-Goog-Colab-Token": xsrf_token}
        )
        self._check_response(resp)
        data = json.loads(_strip_xssi(resp.text))

        # Check outcome
        outcome = data.get("outcome", 0)
        if outcome == 1:
            raise ColabInsufficientQuotaError("Quota denied by Colab", status_code=403)
        if outcome == 2:
            raise ColabInsufficientQuotaError("Compute quota exceeded", status_code=403)
        if outcome == 5:
            raise ColabAPIError("Account is denylisted by Colab", status_code=403)
        if outcome not in (0, 4):
            raise ColabAPIError(f"Unexpected assignment outcome: {outcome}")

        assignment = self._parse_assignment(data)
        logger.info(f"Colab runtime assigned: endpoint={assignment.endpoint}")
        return assignment

    async def unassign_kernel(self, endpoint: str) -> None:
        """Release a Colab runtime using two-step XSRF pattern.

        Args:
            endpoint: Runtime endpoint ID from assignment
        """
        # Step 1: GET to obtain XSRF token
        resp = await self._api_request("GET", f"/tun/m/unassign/{endpoint}")
        self._check_response(resp)
        data = json.loads(_strip_xssi(resp.text))
        xsrf_token = data.get("token", "")

        # Step 2: POST with XSRF token
        if xsrf_token:
            resp = await self._api_request(
                "POST", f"/tun/m/unassign/{endpoint}",
                extra_headers={"X-Goog-Colab-Token": xsrf_token}
            )
            # Don't check response - may be empty
        logger.info(f"Colab runtime unassigned: {endpoint}")

    # ==================== Keep-Alive ====================

    async def keep_alive(self, endpoint: str) -> None:
        """Send keep-alive ping to prevent runtime from idling out.

        Args:
            endpoint: Runtime endpoint ID
        """
        resp = await self._api_request(
            "GET", f"/tun/m/{endpoint}/keep-alive/",
            extra_headers={"X-Colab-Tunnel": "Google"}
        )
        logger.debug(f"Keep-alive sent for {endpoint}, status={resp.status_code}")

    # ==================== Proxy Token Refresh ====================

    async def refresh_proxy_token(self, endpoint: str) -> RuntimeProxyInfo:
        """Refresh the runtime proxy token via GAPI domain.

        Args:
            endpoint: Runtime endpoint ID

        Returns:
            Fresh RuntimeProxyInfo with new token
        """
        resp = await self._gapi_request(
            "GET", f"/v1/runtime-proxy-token?endpoint={endpoint}&port=8080"
        )
        resp.raise_for_status()
        data = resp.json()

        ttl_str = data.get("tokenTtl", "3600s")
        ttl = int(ttl_str.rstrip("s")) if ttl_str else 3600

        return RuntimeProxyInfo(
            token=data["token"],
            url=data.get("url", ""),
            token_expires_seconds=ttl,
        )

    # ==================== Assignments Listing ====================

    async def list_assignments(self) -> List[dict]:
        """List all active runtime assignments via GAPI domain."""
        resp = await self._gapi_request("GET", "/v1/assignments")
        resp.raise_for_status()
        data = resp.json()
        return data.get("assignments", [])

    # ==================== User Info ====================

    async def get_user_info(self) -> dict:
        """Get user info (subscription tier, available accelerators, etc.)."""
        resp = await self._gapi_request("GET", "/v1/user-info")
        resp.raise_for_status()
        return resp.json()

    # ==================== Jupyter Session ====================

    async def create_jupyter_session(self, proxy_url: str, proxy_token: str) -> JupyterSession:
        """Create a Jupyter notebook session on the runtime.

        Must be called after assignment, before WebSocket connection.

        Args:
            proxy_url: Runtime proxy URL from assignment
            proxy_token: Proxy token from assignment

        Returns:
            JupyterSession with session_id and kernel_id
        """
        resp = await self._runtime_request(
            "POST", proxy_url, "/api/sessions", proxy_token,
            json={
                "kernel": {"name": "python3"},
                "name": "colab",
                "path": "colab.ipynb",
                "type": "notebook",
            }
        )
        resp.raise_for_status()
        data = resp.json()

        kernel = data.get("kernel", {})
        session = JupyterSession(
            session_id=data["id"],
            kernel_id=kernel["id"],
            name=data.get("name", "colab"),
        )
        logger.info(f"Jupyter session created: session={session.session_id}, kernel={session.kernel_id}")
        return session

    async def delete_jupyter_session(self, proxy_url: str, proxy_token: str,
                                      session_id: str) -> None:
        """Delete a Jupyter session on the runtime."""
        try:
            await self._runtime_request(
                "DELETE", proxy_url, f"/api/sessions/{session_id}", proxy_token
            )
        except Exception as e:
            logger.warning(f"Failed to delete Jupyter session {session_id}: {e}")

    # ==================== Helpers ====================

    def _check_response(self, resp: httpx.Response) -> None:
        """Check for common error status codes."""
        if resp.status_code == 412:
            raise ColabTooManyAssignmentsError(
                "Too many active Colab runtimes. Disconnect one first.", status_code=412
            )
        if resp.status_code == 403:
            try:
                body = json.loads(_strip_xssi(resp.text))
                msg = body.get("error", {}).get("message", resp.text[:200])
            except (json.JSONDecodeError, KeyError):
                msg = resp.text[:200]
            if "quota" in msg.lower():
                raise ColabInsufficientQuotaError(msg, status_code=403)
            raise ColabAPIError(msg, status_code=403)
        if resp.status_code >= 400:
            raise ColabAPIError(resp.text[:500], status_code=resp.status_code)

    def _parse_assignment(self, data: dict) -> ColabAssignment:
        """Parse an assignment response into ColabAssignment."""
        proxy_data = data.get("runtimeProxyInfo", {})
        ttl = proxy_data.get("tokenExpiresInSeconds",
                             int(proxy_data.get("tokenTtl", "3600s").rstrip("s") or 3600))

        return ColabAssignment(
            endpoint=data.get("endpoint", ""),
            proxy_info=RuntimeProxyInfo(
                token=proxy_data.get("token", ""),
                url=proxy_data.get("url", ""),
                token_expires_seconds=ttl,
            ),
            accelerator=data.get("accelerator", ""),
            outcome=data.get("outcome", 0),
            idle_timeout=data.get("fit", 1800),
        )
