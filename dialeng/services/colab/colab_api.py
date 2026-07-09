"""Async Colab REST API client.

This module follows googlecolab/google-colab-cli semantics while preserving
Dialeng's async architecture:
- tunnel assignment lifecycle via /tun/m/*
- short-timeout tunnel keep-alive
- runtime proxy Jupyter APIs, including Contents API file transfer
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import quote, urlencode

import httpx

logger = logging.getLogger(__name__)

COLAB_API_BASE = "https://colab.research.google.com"
COLAB_GAPI_BASE = "https://colab.pa.googleapis.com"
COLAB_CLIENT_AGENT = "dialeng"
XSSI_PREFIX = ")]}'\n"
KEEP_ALIVE_TIMEOUT_SECONDS = 10.0


class Accelerator(str, Enum):
    NONE = "NONE"
    G4 = "G4"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    V5E1 = "V5E1"
    V6E1 = "V6E1"


class Variant(str, Enum):
    DEFAULT = "DEFAULT"
    GPU = "GPU"
    TPU = "TPU"


def _strip_xssi(response_text: str) -> str:
    """Strip Colab's XSSI protection prefix."""
    if response_text.startswith(XSSI_PREFIX):
        return response_text[len(XSSI_PREFIX):]
    if response_text.startswith(")]}'"):
        return response_text[4:].lstrip("\n")
    return response_text


def _loads_xssi_json(response_text: str) -> dict[str, Any]:
    body = _strip_xssi(response_text)
    return json.loads(body) if body else {}


def _make_notebook_hash() -> str:
    """Generate the web-safe notebook hash format used by Colab assignment."""
    uuid_str = str(uuid.uuid4())
    return uuid_str.replace("-", "_") + "." * (44 - len(uuid_str))


def _normalize_variant(value: str | Variant | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, Variant):
        return value.value
    return str(value)


def _normalize_accelerator(value: str | Accelerator | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, Accelerator):
        return "" if value == Accelerator.NONE else value.value
    return str(value)


@dataclass
class RuntimeProxyInfo:
    """Proxy connection info for a Colab runtime."""

    token: str
    url: str
    token_expires_seconds: int = 3600


@dataclass
class ColabAssignment:
    """Runtime assignment returned by Colab's tunnel frontend."""

    endpoint: str
    proxy_info: RuntimeProxyInfo
    accelerator: str = ""
    variant: str = ""
    outcome: int = 0
    idle_timeout: int = 1800


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


class ColabInsufficientQuotaError(ColabAPIError):
    """User has exceeded their compute quota."""


class ColabAPIClient:
    """Client for Colab tunnel, gapi, and runtime-proxy APIs."""

    def __init__(self, auth_service):
        self.auth = auth_service

    def _base_headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Colab-Client-Agent": COLAB_CLIENT_AGENT,
        }

    async def _api_request(
        self,
        method: str,
        path: str,
        extra_headers: dict[str, str] | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Make authenticated request to colab.research.google.com."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)
        if extra_headers:
            headers.update(extra_headers)

        url = f"{COLAB_API_BASE}{path}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}authuser=0"
        timeout = kwargs.pop("timeout", 30.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    async def _gapi_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make authenticated request to colab.pa.googleapis.com."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)
        timeout = kwargs.pop("timeout", 30.0)

        url = f"{COLAB_GAPI_BASE}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, headers=headers, **kwargs)

    async def _runtime_request(
        self,
        method: str,
        proxy_url: str,
        path: str,
        proxy_token: str,
        **kwargs,
    ) -> httpx.Response:
        """Make request to a Colab runtime proxy."""
        token = await self.auth.get_access_token()
        headers = self._base_headers(token)
        headers["X-Colab-Runtime-Proxy-Token"] = proxy_token
        headers["Content-Type"] = "application/json"
        headers.update(kwargs.pop("headers", {}) or {})

        params = {"authuser": "0", "colab-runtime-proxy-token": proxy_token}
        params.update(kwargs.pop("params", {}) or {})

        timeout = kwargs.pop("timeout", 30.0)
        url = f"{proxy_url.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(
                method,
                url,
                headers=headers,
                params=params,
                **kwargs,
            )

    def _assignment_query(
        self,
        notebook_hash: str,
        variant: str | Variant | None,
        accelerator: str | Accelerator | None,
        shape: str,
    ) -> str:
        params = {"nbh": notebook_hash}
        variant_value = _normalize_variant(variant)
        accelerator_value = _normalize_accelerator(accelerator)
        if variant_value:
            params["variant"] = variant_value
        if accelerator_value:
            params["accelerator"] = accelerator_value
        if shape:
            params["shape"] = shape
        return "?" + urlencode(params)

    async def assign_kernel(
        self,
        variant: str | Variant = "",
        accelerator: str | Accelerator = "",
        shape: str = "",
    ) -> ColabAssignment:
        """Assign or reuse a Colab runtime using the two-step XSRF flow."""
        nbh = _make_notebook_hash()
        query = self._assignment_query(nbh, variant, accelerator, shape)

        resp = await self._api_request("GET", f"/tun/m/assign{query}")
        self._check_response(resp)
        data = _loads_xssi_json(resp.text)

        if "runtimeProxyInfo" in data:
            assignment = self._parse_assignment(data)
            logger.info("Reusing existing Colab runtime assignment (endpoint=%s)", assignment.endpoint)
            return assignment

        xsrf_token = data.get("token")
        if not xsrf_token:
            raise ColabAPIError("No XSRF token in assign GET response")

        resp = await self._api_request(
            "POST",
            f"/tun/m/assign{query}",
            extra_headers={"X-Goog-Colab-Token": xsrf_token},
        )
        self._check_response(resp)
        data = _loads_xssi_json(resp.text)
        self._check_assignment_outcome(data)

        assignment = self._parse_assignment(data)
        logger.info(
            "Colab runtime assigned (endpoint=%s, variant=%s, accelerator=%s)",
            assignment.endpoint,
            assignment.variant,
            assignment.accelerator,
        )
        return assignment

    async def unassign_kernel(self, endpoint: str) -> None:
        """Release a Colab runtime using the two-step XSRF flow."""
        resp = await self._api_request("GET", f"/tun/m/unassign/{endpoint}")
        self._check_response(resp)
        data = _loads_xssi_json(resp.text)
        xsrf_token = data.get("token", "")

        if xsrf_token:
            resp = await self._api_request(
                "POST",
                f"/tun/m/unassign/{endpoint}",
                extra_headers={"X-Goog-Colab-Token": xsrf_token},
            )
            self._check_response(resp)
        logger.info("Colab runtime unassigned: %s", endpoint)

    async def keep_alive(self, endpoint: str) -> None:
        """Refresh the assignment idle timer.

        The tunnel frontend records activity before forwarding to the VM. The
        VM often does not answer this path, so a read timeout is considered a
        successful keep-alive, matching google-colab-cli.
        """
        try:
            resp = await self._api_request(
                "GET",
                f"/tun/m/{endpoint}/keep-alive/",
                extra_headers={"X-Colab-Tunnel": "Google"},
                timeout=KEEP_ALIVE_TIMEOUT_SECONDS,
            )
        except httpx.ReadTimeout:
            logger.debug("Colab keep-alive accepted by tunnel frontend but VM timed out: %s", endpoint)
            return
        self._check_response(resp)
        logger.debug("Colab keep-alive sent for %s, status=%s", endpoint, resp.status_code)

    async def refresh_proxy_token(self, endpoint: str) -> RuntimeProxyInfo:
        """Refresh the runtime proxy token via Colab GAPI."""
        resp = await self._gapi_request(
            "GET", f"/v1/runtime-proxy-token?endpoint={endpoint}&port=8080"
        )
        resp.raise_for_status()
        data = resp.json()

        ttl_raw = data.get("tokenTtl", "3600s")
        ttl = int(str(ttl_raw).rstrip("s")) if ttl_raw else 3600
        return RuntimeProxyInfo(
            token=data["token"],
            url=data.get("url", ""),
            token_expires_seconds=ttl,
        )

    async def list_tunnel_assignments(self) -> list[ColabAssignment]:
        """List active Colab tunnel assignments for the authenticated user."""
        resp = await self._api_request("GET", "/tun/m/assignments")
        self._check_response(resp)
        data = _loads_xssi_json(resp.text)
        return [self._parse_assignment(item) for item in data.get("assignments", [])]

    async def list_assignments(self) -> list[dict[str, Any]]:
        """Backward-compatible assignment list API."""
        return [assignment.__dict__ for assignment in await self.list_tunnel_assignments()]

    async def get_user_info(self) -> dict[str, Any]:
        """Get user info from Colab GAPI."""
        resp = await self._gapi_request("GET", "/v1/user-info")
        resp.raise_for_status()
        return resp.json()

    async def create_jupyter_session(self, proxy_url: str, proxy_token: str) -> JupyterSession:
        """Create a Jupyter session and kernel on the assigned runtime."""
        resp = await self._runtime_request(
            "POST",
            proxy_url,
            "/api/sessions",
            proxy_token,
            json={
                "kernel": {"name": "python3"},
                "name": "colab",
                "path": "colab.ipynb",
                "type": "notebook",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        kernel = data.get("kernel", {})
        session = JupyterSession(
            session_id=data["id"],
            kernel_id=kernel["id"],
            name=data.get("name", "colab"),
        )
        logger.info("Jupyter session created: session=%s, kernel=%s", session.session_id, session.kernel_id)
        return session

    async def delete_jupyter_session(
        self,
        proxy_url: str,
        proxy_token: str,
        session_id: str,
    ) -> None:
        """Delete a Jupyter session on the runtime."""
        try:
            resp = await self._runtime_request(
                "DELETE", proxy_url, f"/api/sessions/{session_id}", proxy_token
            )
            if resp.status_code not in (200, 204, 404):
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to delete Jupyter session %s: %s", session_id, exc)

    async def restart_kernel(self, proxy_url: str, proxy_token: str, kernel_id: str) -> None:
        resp = await self._runtime_request(
            "POST",
            proxy_url,
            f"/api/kernels/{kernel_id}/restart",
            proxy_token,
        )
        resp.raise_for_status()

    async def interrupt_kernel(self, proxy_url: str, proxy_token: str, kernel_id: str) -> None:
        resp = await self._runtime_request(
            "POST",
            proxy_url,
            f"/api/kernels/{kernel_id}/interrupt",
            proxy_token,
        )
        resp.raise_for_status()

    async def contents_mkdir(self, proxy_url: str, proxy_token: str, remote_path: str) -> dict[str, Any]:
        """Create a directory through the runtime's Jupyter Contents API."""
        path = quote(remote_path.strip("/"), safe="/")
        payload = {
            "name": remote_path.rstrip("/").split("/")[-1] if remote_path.strip("/") else "",
            "path": remote_path.strip("/"),
            "type": "directory",
            "content": [],
        }
        resp = await self._runtime_request(
            "PUT",
            proxy_url,
            f"/api/contents/{path}",
            proxy_token,
            json=payload,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        if resp.status_code in (400, 409):
            logger.debug("Ignoring existing Colab Contents directory: %s", remote_path)
            return {}
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def contents_upload(
        self,
        proxy_url: str,
        proxy_token: str,
        remote_path: str,
        content: str | bytes,
    ) -> dict[str, Any]:
        """Upload a text or binary file through the Jupyter Contents API."""
        raw = content.encode("utf-8") if isinstance(content, str) else content
        path = quote(remote_path.strip("/"), safe="/")
        filename = remote_path.rstrip("/").split("/")[-1]
        payload = {
            "name": filename,
            "path": remote_path.strip("/"),
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(raw).decode("ascii"),
            "chunk": 1,
        }
        resp = await self._runtime_request(
            "PUT",
            proxy_url,
            f"/api/contents/{path}",
            proxy_token,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def contents_download(self, proxy_url: str, proxy_token: str, remote_path: str) -> bytes:
        path = quote(remote_path.strip("/"), safe="/")
        resp = await self._runtime_request(
            "GET",
            proxy_url,
            f"/api/contents/{path}",
            proxy_token,
            params={"content": "1"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("type") == "directory":
            raise IsADirectoryError(remote_path)
        content = data.get("content", "")
        if data.get("format") == "base64":
            return base64.b64decode(content)
        return str(content).encode("utf-8")

    async def contents_delete(self, proxy_url: str, proxy_token: str, remote_path: str) -> None:
        path = quote(remote_path.strip("/"), safe="/")
        resp = await self._runtime_request(
            "DELETE", proxy_url, f"/api/contents/{path}", proxy_token
        )
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()

    def _check_response(self, resp: httpx.Response) -> None:
        """Map known Colab HTTP failures to Dialeng errors."""
        if resp.status_code == 412:
            raise ColabTooManyAssignmentsError(
                "Too many active Colab runtimes. Disconnect one first.",
                status_code=412,
            )
        if resp.status_code == 403:
            msg = resp.text[:500]
            try:
                body = _loads_xssi_json(resp.text)
                msg = body.get("error", {}).get("message", msg)
            except Exception:
                pass
            if "quota" in msg.lower():
                raise ColabInsufficientQuotaError(msg, status_code=403)
            raise ColabAPIError(msg, status_code=403)
        if resp.status_code >= 400:
            raise ColabAPIError(resp.text[:500], status_code=resp.status_code)

    def _check_assignment_outcome(self, data: dict[str, Any]) -> None:
        outcome = data.get("outcome", 0)
        if outcome == 1:
            raise ColabInsufficientQuotaError("Quota denied by Colab", status_code=403)
        if outcome == 2:
            raise ColabInsufficientQuotaError("Compute quota exceeded", status_code=403)
        if outcome == 5:
            raise ColabAPIError("Account is denylisted by Colab", status_code=403)
        if outcome not in (0, 4, None):
            raise ColabAPIError(f"Unexpected assignment outcome: {outcome}")

    def _parse_assignment(self, data: dict[str, Any]) -> ColabAssignment:
        proxy_data = data.get("runtimeProxyInfo") or data.get("runtime_proxy_info") or {}
        ttl = proxy_data.get("tokenExpiresInSeconds")
        if ttl is None:
            ttl_raw = proxy_data.get("tokenTtl", "3600s")
            ttl = int(str(ttl_raw).rstrip("s") or 3600)

        variant = data.get("variant", "")
        if isinstance(variant, int):
            variant = {0: Variant.DEFAULT.value, 1: Variant.GPU.value, 2: Variant.TPU.value}.get(variant, str(variant))

        return ColabAssignment(
            endpoint=data.get("endpoint", ""),
            proxy_info=RuntimeProxyInfo(
                token=proxy_data.get("token", ""),
                url=proxy_data.get("url", ""),
                token_expires_seconds=int(ttl or 3600),
            ),
            accelerator=data.get("accelerator", ""),
            variant=variant or "",
            outcome=data.get("outcome", 0),
            idle_timeout=data.get("fit", data.get("idleTimeout", 1800)),
        )
