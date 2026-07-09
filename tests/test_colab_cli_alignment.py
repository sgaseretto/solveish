"""Tests for google-colab-cli alignment behavior."""

import asyncio
import base64
import time
from urllib.parse import parse_qs, urlparse

import httpx

from dialeng.services.colab.colab_api import (
    Accelerator,
    COLAB_API_BASE,
    KEEP_ALIVE_TIMEOUT_SECONDS,
    RuntimeProxyInfo,
    ColabAPIClient,
    ColabAssignment,
    ColabAPIError,
    ColabInsufficientQuotaError,
    ColabTooManyAssignmentsError,
    JupyterSession,
    Variant,
    XSSI_PREFIX,
    _make_notebook_hash,
    _loads_xssi_json,
)
from dialeng.services.colab.colab_auth import (
    ColabAuthProvider,
    ColabAuthService,
    OAuthClientCredentials,
    REMOTE_REDIRECT_URI,
    resolve_oauth_credentials,
)
from dialeng.services.colab.colab_kernel import ColabKernel, JUPYTER_WS_DEFAULT_SUBPROTOCOL
from dialeng.services.colab.colab_state import ColabSessionMetadata, ColabSessionStore
from dialeng.services.colab.colab_terminal import build_colab_tty_ws_url
from dialeng.services.dialeng_config import _parse_config


class _Auth:
    async def get_access_token(self):
        return "access-token"


def _response(status_code: int, text: str = "", *, json_body=None) -> httpx.Response:
    request = httpx.Request("GET", f"{COLAB_API_BASE}/test")
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text, request=request)


def test_colab_xssi_json_and_notebook_hash_format():
    parsed = _loads_xssi_json(XSSI_PREFIX + '{"ok": true}')
    assert parsed == {"ok": True}

    parsed_without_newline = _loads_xssi_json(")]}'" + '{"ok": true}')
    assert parsed_without_newline == {"ok": True}

    notebook_hash = _make_notebook_hash()
    assert len(notebook_hash) == 44
    assert "-" not in notebook_hash
    assert "_" in notebook_hash


def test_colab_assignment_parses_cli_shapes_and_variant_ints():
    client = ColabAPIClient(_Auth())
    assignment = client._parse_assignment({
        "endpoint": "runtime-endpoint",
        "runtimeProxyInfo": {
            "token": "proxy-token",
            "url": "https://runtime.proxy",
            "tokenExpiresInSeconds": 120,
        },
        "accelerator": Accelerator.T4.value,
        "variant": 1,
        "fit": 900,
    })

    assert assignment.endpoint == "runtime-endpoint"
    assert assignment.proxy_info.token == "proxy-token"
    assert assignment.proxy_info.url == "https://runtime.proxy"
    assert assignment.proxy_info.token_expires_seconds == 120
    assert assignment.variant == Variant.GPU.value
    assert assignment.accelerator == Accelerator.T4.value
    assert assignment.idle_timeout == 900

    snake_assignment = client._parse_assignment({
        "endpoint": "runtime-snake",
        "runtime_proxy_info": {
            "token": "snake-token",
            "url": "https://snake.proxy",
            "tokenTtl": "55s",
        },
        "idleTimeout": 123,
    })
    assert snake_assignment.proxy_info.token_expires_seconds == 55
    assert snake_assignment.idle_timeout == 123


def test_colab_response_and_assignment_outcome_errors():
    client = ColabAPIClient(_Auth())

    try:
        client._check_response(_response(412, "too many"))
    except ColabTooManyAssignmentsError as exc:
        assert exc.status_code == 412
    else:
        raise AssertionError("412 should raise ColabTooManyAssignmentsError")

    try:
        client._check_response(_response(403, XSSI_PREFIX + '{"error": {"message": "quota exceeded"}}'))
    except ColabInsufficientQuotaError as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("quota 403 should raise ColabInsufficientQuotaError")

    try:
        client._check_assignment_outcome({"outcome": 5})
    except ColabAPIError as exc:
        assert "denylisted" in str(exc)
    else:
        raise AssertionError("denylist outcome should raise ColabAPIError")


def test_colab_assignment_query_uses_runtime_variant_and_accelerator():
    client = ColabAPIClient(_Auth())
    query = client._assignment_query("notebook", Variant.GPU, Accelerator.T4, "")
    parsed = parse_qs(urlparse(f"https://example.test/{query}").query)

    assert parsed["nbh"] == ["notebook"]
    assert parsed["variant"] == ["GPU"]
    assert parsed["accelerator"] == ["T4"]


def test_colab_list_tunnel_assignments_parses_xssi_assignment_list():
    client = ColabAPIClient(_Auth())

    async def fake_api_request(method, path, **kwargs):
        assert method == "GET"
        assert path == "/tun/m/assignments"
        return _response(200, XSSI_PREFIX + """{
            "assignments": [{
                "endpoint": "runtime-1",
                "runtimeProxyInfo": {
                    "token": "token-1",
                    "url": "https://runtime.proxy",
                    "tokenExpiresInSeconds": 60
                }
            }]
        }""")

    client._api_request = fake_api_request

    assignments = asyncio.run(client.list_tunnel_assignments())

    assert len(assignments) == 1
    assert assignments[0].endpoint == "runtime-1"
    assert assignments[0].proxy_info.token == "token-1"


def test_colab_keep_alive_treats_read_timeout_as_success_and_sets_cli_headers():
    client = ColabAPIClient(_Auth())
    calls = []

    async def fake_api_request(method, path, extra_headers=None, **kwargs):
        calls.append((method, path, extra_headers, kwargs))
        raise httpx.ReadTimeout("runtime did not respond")

    client._api_request = fake_api_request

    asyncio.run(client.keep_alive("runtime-1"))

    assert calls == [
        (
            "GET",
            "/tun/m/runtime-1/keep-alive/",
            {"X-Colab-Tunnel": "Google"},
            {"timeout": KEEP_ALIVE_TIMEOUT_SECONDS},
        )
    ]


def test_colab_keep_alive_preserves_real_http_failures():
    client = ColabAPIClient(_Auth())

    async def fake_api_request(method, path, extra_headers=None, **kwargs):
        return _response(404, "missing")

    client._api_request = fake_api_request

    try:
        asyncio.run(client.keep_alive("runtime-1"))
    except ColabAPIError as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("404 keep-alive should be a real failure")


def test_colab_contents_upload_uses_base64_payload_and_proxy_token_params():
    client = ColabAPIClient(_Auth())
    calls = []

    async def fake_runtime_request(method, proxy_url, path, proxy_token, **kwargs):
        calls.append((method, proxy_url, path, proxy_token, kwargs))
        return _response(200, json_body={"path": "pkg/data.bin"})

    client._runtime_request = fake_runtime_request

    result = asyncio.run(
        client.contents_upload("https://runtime.proxy", "proxy-token", "pkg/data.bin", b"\x00\xff")
    )

    assert result == {"path": "pkg/data.bin"}
    method, proxy_url, path, proxy_token, kwargs = calls[0]
    assert method == "PUT"
    assert proxy_url == "https://runtime.proxy"
    assert path == "/api/contents/pkg/data.bin"
    assert proxy_token == "proxy-token"
    assert kwargs["json"]["format"] == "base64"
    assert kwargs["json"]["content"] == base64.b64encode(b"\x00\xff").decode("ascii")


def test_colab_contents_download_decodes_base64_and_rejects_directories():
    client = ColabAPIClient(_Auth())

    async def fake_runtime_request(method, proxy_url, path, proxy_token, **kwargs):
        return _response(200, json_body={
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(b"hello").decode("ascii"),
        })

    client._runtime_request = fake_runtime_request

    assert asyncio.run(client.contents_download("https://runtime.proxy", "token", "file.txt")) == b"hello"

    async def fake_directory_request(method, proxy_url, path, proxy_token, **kwargs):
        return _response(200, json_body={"type": "directory", "content": []})

    client._runtime_request = fake_directory_request

    try:
        asyncio.run(client.contents_download("https://runtime.proxy", "token", "folder"))
    except IsADirectoryError:
        pass
    else:
        raise AssertionError("directory contents should raise IsADirectoryError")


def test_colab_kernel_sync_project_files_uses_contents_api_without_execute_request():
    class FakeAPI:
        def __init__(self):
            self.mkdirs = []
            self.uploads = []

        async def contents_mkdir(self, proxy_url, proxy_token, remote_path):
            self.mkdirs.append((proxy_url, proxy_token, remote_path))
            return {}

        async def contents_upload(self, proxy_url, proxy_token, remote_path, content):
            self.uploads.append((proxy_url, proxy_token, remote_path, content))
            return {}

    api = FakeAPI()
    kernel = ColabKernel(api_client=api)
    kernel._assignment = ColabAssignment(
        endpoint="runtime-1",
        proxy_info=RuntimeProxyInfo(token="proxy-token", url="https://runtime.proxy"),
    )
    kernel._jupyter_session = JupyterSession(session_id="session-1", kernel_id="kernel-1")

    async def fail_run_setup_code(*args, **kwargs):
        raise AssertionError("Contents API sync must not use execute_request")

    kernel.run_setup_code = fail_run_setup_code

    result = asyncio.run(kernel.sync_project_files([
        ("pkg/mod.py", "x = 1\n"),
        ("pkg/assets/data.bin", b"\x00\x01"),
    ]))

    assert result["status"] == "ok"
    assert result["transfer_method"] == "contents_api"
    assert result["file_count"] == 2
    assert result["total_bytes"] == len("x = 1\n".encode("utf-8")) + 2
    assert [item[2] for item in api.mkdirs] == ["pkg", "pkg/assets"]
    assert [item[2] for item in api.uploads] == ["pkg/mod.py", "pkg/assets/data.bin"]


def test_colab_session_store_save_load_prune_and_corrupt_fallback(tmp_path):
    path = tmp_path / "sessions.json"
    store = ColabSessionStore(path)
    metadata = ColabSessionMetadata(
        notebook_id="nb",
        endpoint="runtime-1",
        proxy_url="https://runtime.proxy",
        proxy_token_expires_at=123.0,
        runtime_type="gpu",
        accelerator="T4",
        kernel_id="kernel-1",
        jupyter_session_id="session-1",
        connection_state="connected",
        last_activity_at=456.0,
    )

    store.upsert(metadata)
    loaded = store.load_all()["nb"]
    assert loaded.endpoint == "runtime-1"
    assert loaded.kernel_id == "kernel-1"
    assert loaded.jupyter_session_id == "session-1"
    assert loaded.runtime_type == "gpu"

    assert store.prune_stale_endpoints({"other-runtime"}) == 1
    assert store.load_all() == {}

    path.write_text("{not json", encoding="utf-8")
    assert store.load_all() == {}


def test_colab_kernel_restores_persisted_assignment_metadata(tmp_path):
    class FakeAPI:
        async def refresh_proxy_token(self, endpoint):
            assert endpoint == "runtime-1"
            return RuntimeProxyInfo(token="fresh-token", url="https://runtime.proxy", token_expires_seconds=99)

    store = ColabSessionStore(tmp_path / "sessions.json")
    store.upsert(ColabSessionMetadata(
        notebook_id="nb",
        endpoint="runtime-1",
        proxy_url="https://old.proxy",
        proxy_token_expires_at=time.time() - 1,
        runtime_type="cpu",
        kernel_id="kernel-1",
        jupyter_session_id="session-1",
    ))
    kernel = ColabKernel(api_client=FakeAPI(), notebook_id="nb", runtime_type="cpu")
    kernel._session_store = store

    restored = asyncio.run(kernel._restore_persisted_assignment())

    assert restored is True
    assert kernel._assignment.endpoint == "runtime-1"
    assert kernel._assignment.proxy_info.token == "fresh-token"
    assert kernel._jupyter_session.kernel_id == "kernel-1"


def test_colab_auth_provider_selection_and_remote_url(monkeypatch, tmp_path):
    from dialeng.services.colab import colab_auth as auth_mod

    token_dir = tmp_path / ".dialeng"
    monkeypatch.setattr(auth_mod, "TOKEN_DIR", token_dir)
    monkeypatch.setattr(auth_mod, "TOKEN_FILE", token_dir / "colab_tokens.json")

    creds = OAuthClientCredentials("client-id", "client-secret", source="test")
    service = ColabAuthService(credentials=creds, auth_provider=ColabAuthProvider.OAUTH_REMOTE.value)
    url = service.get_remote_auth_url(state="state-1")
    parsed = parse_qs(urlparse(url).query)

    assert parsed["redirect_uri"] == [REMOTE_REDIRECT_URI]
    assert parsed["token_usage"] == ["remote"]
    assert parsed["state"] == ["state-1"]

    adc_creds = asyncio.run(resolve_oauth_credentials(ColabAuthProvider.ADC.value))
    assert adc_creds.source == "adc"

    monkeypatch.setenv("DIALENG_COLAB_AUTH_PROVIDER", "adc")
    config = _parse_config({"colab": {"enabled": True, "auth_provider": "oauth_local"}})
    assert config.colab_auth_provider == "adc"


def test_colab_tty_url_helper_codifies_future_terminal_path():
    url = build_colab_tty_ws_url("https://runtime.proxy/base", "proxy-token")
    parsed = urlparse(url)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "runtime.proxy"
    assert parsed.path == "/colab/tty"
    assert parse_qs(parsed.query)["colab-runtime-proxy-token"] == ["proxy-token"]


def test_colab_kernel_websocket_uses_jupyter_subprotocol_constant():
    assert JUPYTER_WS_DEFAULT_SUBPROTOCOL == "default"
    assert hasattr(ColabKernel, "_WS_CLOSE_TYPES")
