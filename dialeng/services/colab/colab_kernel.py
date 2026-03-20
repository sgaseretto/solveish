"""Colab kernel implementation using Jupyter wire protocol over WebSocket.

Connects to a Google Colab runtime via:
1. REST API to assign a runtime (two-step XSRF pattern)
2. REST API to create a Jupyter session (gets kernel ID)
3. WebSocket to runtime proxy for Jupyter wire protocol (execute, interrupt, etc.)
4. HTTP keep-alive pings every 5 minutes
5. Proxy token refresh before expiry

Lifecycle: assign → create session → connect WS → execute → keep-alive → disconnect
"""
import asyncio
import json
import uuid
import logging
import time
from typing import AsyncIterator, Optional

import aiohttp

from dialeng.document.cell import CellOutput
from dialeng.services.kernel.base_kernel import BaseKernel, KernelInfo, KernelStatus
from .colab_api import ColabAPIClient, ColabAssignment, JupyterSession

logger = logging.getLogger(__name__)

KEEP_ALIVE_INTERVAL = 300  # 5 minutes
TOKEN_REFRESH_BUFFER = 300  # Refresh proxy token 5 min before expiry
WS_RECEIVE_LOG_TIMEOUT = 60  # Log when Colab is quiet for a long time

# ============================================================================
# DialogHelper proxy protocol constants
# ============================================================================
# When Colab code calls dialoghelper functions, the monkey-patched call_endp
# uses input() with these magic prefixes to tunnel HTTP requests through
# Jupyter's stdin channel back to Dialeng.
DH_PROXY_PREFIX = "__DH_PROXY__:"
DH_PROXY_RESP_PREFIX = "__DH_PROXY_RESP__:"
DH_PROXY_ERR_PREFIX = "__DH_PROXY_ERR__:"

# Auto-install dialoghelper on Colab
DIALOGHELPER_INSTALL = "%pip install -q dialoghelper"

# Monkey-patch code injected into Colab kernel to replace HTTP transport
# with stdin-based proxy. _prep_endp still runs on Colab (resolves dname,
# assembles data dict), but the actual HTTP call is proxied through Dialeng.
DIALOGHELPER_PROXY_SETUP = '''
try:
    import dialoghelper.core as _dhc
    import json as _json

    class _ProxyResponse:
        """Shim matching httpx.Response interface for _handle_resp."""
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text
        def json(self): return _json.loads(self.text)
        def raise_for_status(self):
            if self.status_code >= 400: raise Exception(f"HTTP {self.status_code}: {self.text}")

    def _proxy_call(path, data, headers):
        """Send HTTP request via input() proxy, return Response-like object."""
        request = _json.dumps({
            "path": path,
            "data": {k: str(v) if not isinstance(v, str) else v for k, v in data.items()},
            "headers": dict(headers),
        })
        raw_reply = input("__DH_PROXY__:" + request)
        if raw_reply.startswith("__DH_PROXY_RESP__:"):
            resp = _json.loads(raw_reply[len("__DH_PROXY_RESP__:"):])
            return _ProxyResponse(resp.get("status", 200), resp.get("body", ""))
        elif raw_reply.startswith("__DH_PROXY_ERR__:"):
            err = _json.loads(raw_reply[len("__DH_PROXY_ERR__:"):])
            raise ConnectionError(f"Proxy error: {err.get('error', 'unknown')}")
        else:
            raise ConnectionError(f"Unexpected proxy response: {raw_reply[:200]}")

    def _patched_call_endp(path, dname='', json=False, raiseex=False, id=None, **data):
        url, data, headers = _dhc._prep_endp(path, dname, json, id, data)
        return _dhc._handle_resp(_proxy_call(path, data, headers), json, raiseex)

    async def _patched_call_endpa(path, dname='', json=False, raiseex=False, id=None, **data):
        import asyncio
        url, data, headers = _dhc._prep_endp(path, dname, json, id, data)
        res = await asyncio.get_event_loop().run_in_executor(
            None, _proxy_call, path, data, headers)
        return _dhc._handle_resp(res, json, raiseex)

    _dhc.call_endp = _patched_call_endp
    _dhc.call_endpa = _patched_call_endpa
    print("dialoghelper proxy: active")
except ImportError:
    import sys
    print("dialoghelper proxy: package not found after install", file=sys.stderr)
except Exception as _e:
    import sys
    print(f"dialoghelper proxy: setup failed: {_e}", file=sys.stderr)
'''


class ColabKernel(BaseKernel):
    """Kernel running on Google Colab via WebSocket.

    Implements BaseKernel using Jupyter wire protocol over WebSocket
    to communicate with a remote Colab runtime.
    """

    # Close-related WebSocket message types
    _WS_CLOSE_TYPES = frozenset({
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSING,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.ERROR,
    })

    def __init__(self, api_client: ColabAPIClient, runtime_type: str = "cpu",
                 dialeng_port: int = 8000):
        self._api = api_client
        self._assignment: Optional[ColabAssignment] = None
        self._jupyter_session: Optional[JupyterSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._execution_count: int = 0
        self._is_busy: bool = False
        self._keep_alive_task: Optional[asyncio.Task] = None
        self._token_refresh_task: Optional[asyncio.Task] = None
        self._connection_state: str = "disconnected"  # disconnected|connecting|connected
        self._current_msg_id: Optional[str] = None
        self._token_expires_at: float = 0  # Unix timestamp when proxy token expires
        # Consistent session ID for all Jupyter wire protocol messages
        self._session_id: str = uuid.uuid4().hex
        # Runtime type: "cpu", "gpu", "tpu"
        self.runtime_type: str = runtime_type
        # Dialeng server port for proxying dialoghelper HTTP calls
        self._dialeng_port: int = dialeng_port
        # Current init step description (for status reporting to UI)
        self._init_status: str = ""
        self._keep_alive_failures: int = 0
        self._token_refresh_failures: int = 0

    def _make_header(self, msg_type: str, msg_id: str = None) -> dict:
        """Build a Jupyter wire protocol message header with consistent session ID."""
        return {
            "msg_id": msg_id or uuid.uuid4().hex,
            "msg_type": msg_type,
            "username": "dialeng",
            "session": self._session_id,
            "date": "",
            "version": "5.3",
        }

    @property
    def connection_state(self) -> str:
        return self._connection_state

    async def assign_and_connect(self, max_retries: int = 5) -> None:
        """Assign a Colab runtime, create session, and connect WebSocket.

        Uses self.runtime_type to determine the variant (cpu/gpu/tpu).
        Retries transient errors (503 Service Unavailable) with exponential
        backoff — GPU/TPU runtimes can take time to spin up.
        """
        self._connection_state = "connecting"
        # Map runtime_type to Colab API variant + accelerator parameters
        _RUNTIME_MAP = {
            "cpu":  {"variant": "", "accelerator": ""},
            "gpu":  {"variant": "GPU", "accelerator": "T4"},
            "tpu":  {"variant": "TPU", "accelerator": ""},
        }
        rt = _RUNTIME_MAP.get(self.runtime_type, _RUNTIME_MAP["cpu"])

        from dialeng.services.colab.colab_api import ColabAPIError

        last_error = None
        previous_runtime = self._assignment.endpoint if self._assignment else None
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Assigning Colab runtime (attempt %s/%s, runtime_type=%s)",
                    attempt + 1, max_retries, self.runtime_type,
                )

                # 1. Assign runtime via REST API (two-step XSRF)
                self._assignment = await self._api.assign_kernel(
                    variant=rt["variant"], accelerator=rt["accelerator"]
                )
                proxy = self._assignment.proxy_info
                self._token_expires_at = time.time() + proxy.token_expires_seconds
                self._keep_alive_failures = 0
                self._token_refresh_failures = 0

                if previous_runtime and previous_runtime != self._assignment.endpoint:
                    logger.warning(
                        "Colab runtime replaced (previous=%s, current=%s)",
                        previous_runtime, self._assignment.endpoint,
                    )

                # 2. Create Jupyter session on the runtime
                self._jupyter_session = await self._api.create_jupyter_session(
                    proxy.url, proxy.token
                )

                # 3. Open WebSocket to runtime
                await self._connect_websocket()

                # 4. Wait for kernel to be ready (kernel_info handshake)
                await self._wait_for_kernel_ready()

                # 5. Initialize kernel (matplotlib inline backend, etc.)
                await self._initialize_kernel()

                # 6. Start background tasks
                self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())

                self._connection_state = "connected"
                logger.info(
                    "Connected to Colab runtime: endpoint=%s kernel=%s",
                    self._assignment.endpoint,
                    self._jupyter_session.kernel_id,
                )
                return  # Success
            except ColabAPIError as e:
                last_error = e
                # Retry on transient server errors (503, 500, 502, 504)
                if getattr(e, 'status_code', 0) >= 500 and attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                    logger.warning(
                        f"Colab API returned {e.status_code}, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Non-retryable API error or max retries exhausted
                self._connection_state = "disconnected"
                logger.error(f"Failed to connect to Colab: {e}")
                raise
            except Exception as e:
                self._connection_state = "disconnected"
                logger.exception("Failed to connect to Colab")
                raise

        # Should not reach here, but just in case
        self._connection_state = "disconnected"
        raise last_error

    async def _connect_websocket(self) -> None:
        """Open WebSocket connection to Colab runtime's Jupyter kernel."""
        if not self._assignment or not self._jupyter_session:
            raise RuntimeError("No assignment/session - call assign_and_connect() first")

        proxy = self._assignment.proxy_info
        # Convert https:// to wss://
        ws_base = proxy.url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
        ws_url = (
            f"{ws_base}/api/kernels/{self._jupyter_session.kernel_id}"
            f"/channels?session_id={self._session_id}"
        )

        self._ws_session = aiohttp.ClientSession()
        self._ws = await self._ws_session.ws_connect(
            ws_url,
            headers={
                "X-Colab-Runtime-Proxy-Token": proxy.token,
                "X-Colab-Client-Agent": "dialeng",
            },
            # No heartbeat - Colab's proxy may not handle WebSocket pings.
            # Connection is kept alive via HTTP keep-alive pings instead.
        )
        logger.info(
            "WebSocket connected to Colab runtime proxy (endpoint=%s, kernel=%s)",
            self._assignment.endpoint if self._assignment else None,
            self._jupyter_session.kernel_id if self._jupyter_session else None,
        )

    async def _wait_for_kernel_ready(self, timeout: float = 30.0) -> None:
        """Send kernel_info_request and wait for reply to confirm kernel is ready.

        The Colab VS Code extension does this with a 30-second timeout.
        Without this handshake, the kernel may not be ready for execute requests.
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        msg_id = uuid.uuid4().hex
        kernel_info_msg = {
            "header": self._make_header("kernel_info_request", msg_id),
            "parent_header": {},
            "metadata": {},
            "content": {},
            "channel": "shell",
        }
        await self._ws.send_json(kernel_info_msg)
        logger.info("Sent kernel_info_request, waiting for kernel to be ready...")

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            if loop.time() > deadline:
                logger.warning("Kernel readiness check timed out")
                break

            ws_msg = await self._ws.receive(timeout=timeout)

            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(ws_msg.data)
                msg_type = data.get("header", {}).get("msg_type", "")
                parent_id = data.get("parent_header", {}).get("msg_id")

                if msg_type == "kernel_info_reply" and parent_id == msg_id:
                    logger.info("Colab kernel is ready")
                    return

            elif ws_msg.type in self._WS_CLOSE_TYPES:
                raise RuntimeError(f"WebSocket closed during kernel readiness check: {ws_msg}")

        logger.warning("kernel_info_reply not received, proceeding anyway")

    async def _run_init_code(self, code: str, description: str,
                            timeout: float = 30.0) -> None:
        """Execute init code on kernel, log output, wait for completion."""
        if not self._ws:
            return
        msg_id = uuid.uuid4().hex
        execute_msg = {
            "header": self._make_header("execute_request", msg_id),
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": True,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": False,
            },
            "channel": "shell",
        }
        await self._ws.send_json(execute_msg)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        saw_execute_reply = False
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Colab init timed out: %s (saw_execute_reply=%s)",
                    description, saw_execute_reply,
                )
                break
            ws_msg = await self._ws.receive(timeout=remaining)
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(ws_msg.data)
                parent_id = data.get("parent_header", {}).get("msg_id")
                if parent_id != msg_id:
                    continue
                msg_type = data.get("header", {}).get("msg_type", "")
                if msg_type == "stream":
                    text = data.get("content", {}).get("text", "")
                    if text.strip():
                        logger.info(f"Colab init [{description}]: {text.strip()[:200]}")
                elif msg_type == "error":
                    content = data.get("content", {})
                    logger.warning(
                        f"Colab init [{description}] error: "
                        f"{content.get('ename')}: {content.get('evalue')}"
                    )
                elif msg_type == "execute_reply":
                    saw_execute_reply = True
                elif msg_type == "status":
                    execution_state = data.get("content", {}).get("execution_state")
                    if execution_state == "idle":
                        logger.info("Colab init complete: %s", description)
                        break
            elif ws_msg.type in self._WS_CLOSE_TYPES:
                logger.warning(f"WebSocket closed during init: {description}")
                break

    async def _initialize_kernel(self) -> None:
        """Run setup code on kernel: matplotlib, dialoghelper install, proxy setup.

        Each step runs as a separate execute_request with logging and timeout.
        Progress is exposed via self._init_status for UI display.
        """
        if not self._ws:
            return

        init_steps = [
            ("Setting up matplotlib", "%matplotlib inline", 30.0),
            ("Installing dialoghelper", DIALOGHELPER_INSTALL, 120.0),
            ("Configuring dialoghelper proxy", DIALOGHELPER_PROXY_SETUP, 30.0),
        ]
        for description, code, timeout in init_steps:
            self._init_status = description
            logger.info(f"Colab init: {description}")
            await self._run_init_code(code, description, timeout=timeout)

        self._init_status = ""
        logger.info("Colab kernel initialization complete")

    async def _handle_dh_proxy(self, prompt: str) -> str:
        """Handle a dialoghelper proxy request from Colab kernel.

        Parses request from input prompt, makes local HTTP call to Dialeng,
        returns response formatted for the Colab-side proxy shim.
        """
        try:
            request = json.loads(prompt[len(DH_PROXY_PREFIX):])
            path = request["path"]
            data = request.get("data", {})
            headers = request.get("headers", {})
            url = f"http://localhost:{self._dialeng_port}/{path}"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as resp:
                    body = await resp.text()
                    content_type = resp.headers.get("Content-Type", "text/plain")
                    return DH_PROXY_RESP_PREFIX + json.dumps({
                        "status": resp.status, "body": body,
                        "content_type": content_type,
                    })
        except Exception as e:
            logger.error(f"DialogHelper proxy error: {e}")
            return DH_PROXY_ERR_PREFIX + json.dumps({"error": str(e)})

    async def execute_streaming(
        self,
        code: str,
        notebook_id: str = "",
        cell_id: str = ""
    ) -> AsyncIterator[CellOutput]:
        """Execute code on Colab runtime and stream outputs.

        Sends an execute_request via Jupyter wire protocol and yields
        CellOutput objects for each response message until status:idle.

        We wait for status:idle (the last IOPub message) rather than
        execute_reply (Shell) because Colab multiplexes both channels
        onto a single WebSocket and execute_reply can arrive before
        late IOPub messages like display_data from matplotlib plots.
        """
        if not self._ws or self._ws.closed or self._connection_state not in {"connected", "degraded"}:
            await self.assign_and_connect()

        # Inject dialoghelper magic variables into the remote kernel namespace.
        # The subprocess kernel does this via shell.user_ns, but for Colab we
        # prepend silent assignment code so dialoghelper's find_var() can
        # locate __dialog_name and __msg_id in the call stack.
        if notebook_id or cell_id:
            preamble_parts = []
            if notebook_id:
                preamble_parts.append(f"__dialog_name = {notebook_id!r}")
            if cell_id:
                preamble_parts.append(f"__msg_id = {cell_id!r}")
            code = "\n".join(preamble_parts) + "\n" + code

        msg_id = uuid.uuid4().hex
        self._current_msg_id = msg_id
        self._is_busy = True

        # Send execute_request (Jupyter wire protocol v5.3)
        execute_msg = {
            "header": self._make_header("execute_request", msg_id),
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": True,   # Needed for dialoghelper stdin proxy
                "stop_on_error": True,
            },
            "channel": "shell",
        }
        try:
            await self._ws.send_json(execute_msg)
        except (ConnectionResetError, ConnectionError, OSError) as e:
            # Connection dropped, reconnect and retry
            logger.warning(f"WebSocket send failed ({e}), reconnecting...")
            self._connection_state = "disconnected"
            await self.assign_and_connect()
            msg_id = uuid.uuid4().hex
            self._current_msg_id = msg_id
            execute_msg["header"] = self._make_header("execute_request", msg_id)
            await self._ws.send_json(execute_msg)

        # Read messages until status:idle on IOPub.
        #
        # Colab multiplexes Shell and IOPub onto a single WebSocket, so
        # execute_reply (Shell) may arrive BEFORE display_data (IOPub).
        # Breaking on execute_reply would miss late-arriving rich outputs
        # like matplotlib plots.  Instead we break on status:idle, which
        # is the *last* IOPub message and guarantees all outputs have been
        # delivered.  We also capture execution_count from execute_reply
        # when it arrives (without breaking).
        try:
            while True:
                try:
                    ws_msg = await self._ws.receive(timeout=WS_RECEIVE_LOG_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        "No Colab WebSocket activity for %ss while waiting on cell %s",
                        WS_RECEIVE_LOG_TIMEOUT, cell_id or "<unknown>",
                    )
                    continue

                if ws_msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(ws_msg.data)

                    # Only process messages for our request
                    parent_msg_id = data.get("parent_header", {}).get("msg_id")
                    if parent_msg_id != msg_id:
                        continue

                    msg_type = data.get("header", {}).get("msg_type", "")
                    channel = data.get("channel", "")
                    content = data.get("content", {})

                    logger.debug(
                        f"WS msg: type={msg_type} channel={channel} "
                        f"content_keys={list(content.keys())[:5]}"
                    )

                    if msg_type == "stream":
                        yield CellOutput(
                            output_type="stream",
                            content=content.get("text", ""),
                            stream_name=content.get("name", "stdout"),
                        )

                    elif msg_type in ("display_data", "update_display_data"):
                        display_id = content.get("transient", {}).get("display_id")
                        yield CellOutput(
                            output_type="display_data" if msg_type == "display_data" else "update_display_data",
                            content=content.get("data", {}),
                            metadata=content.get("metadata"),
                            display_id=display_id,
                        )

                    elif msg_type == "execute_result":
                        data_content = content.get("data", {})
                        # If result contains rich MIME types (images, HTML),
                        # treat as display_data so they get rendered properly
                        _RICH_MIMES = {"image/png", "image/jpeg", "image/svg+xml", "text/html", "image/gif"}
                        if _RICH_MIMES & set(data_content.keys()):
                            yield CellOutput(
                                output_type="display_data",
                                content=data_content,
                                metadata=content.get("metadata"),
                            )
                        else:
                            yield CellOutput(
                                output_type="execute_result",
                                content=data_content.get("text/plain", ""),
                                metadata=content.get("metadata"),
                            )

                    elif msg_type == "clear_output":
                        yield CellOutput(
                            output_type="clear_output",
                            content=content.get("wait", False),
                        )

                    elif msg_type == "error":
                        yield CellOutput(
                            output_type="error",
                            ename=content.get("ename", "Error"),
                            evalue=content.get("evalue", ""),
                            traceback=content.get("traceback", []),
                        )

                    elif msg_type == "execute_reply":
                        # Capture execution_count but do NOT break yet —
                        # display_data messages may still be in flight on IOPub.
                        self._execution_count = content.get(
                            "execution_count", self._execution_count + 1
                        )

                    elif msg_type == "status":
                        execution_state = content.get("execution_state")
                        self._is_busy = execution_state == "busy"
                        if execution_state == "idle":
                            # Last IOPub message — all outputs delivered
                            break

                    elif msg_type == "input_request":
                        # Handle stdin — used by dialoghelper proxy and input()
                        prompt = content.get("prompt", "")
                        if prompt.startswith(DH_PROXY_PREFIX):
                            reply_value = await self._handle_dh_proxy(prompt)
                        else:
                            # Non-proxy input() — not supported on remote
                            reply_value = ""

                        input_reply = {
                            "header": self._make_header("input_reply"),
                            "parent_header": data.get("header", {}),
                            "metadata": {},
                            "content": {"value": reply_value, "status": "ok"},
                            "channel": "stdin",
                        }
                        await self._ws.send_json(input_reply)

                elif ws_msg.type in self._WS_CLOSE_TYPES:
                    logger.warning(f"WebSocket close frame received: type={ws_msg.type}")
                    yield CellOutput(
                        output_type="error",
                        ename="ColabConnectionError",
                        evalue="WebSocket connection to Colab lost",
                        traceback=["Connection to Colab runtime was lost"],
                    )
                    self._connection_state = "disconnected"
                    break
                else:
                    # Ignore BINARY, PING, PONG etc.
                    logger.debug(f"Ignoring WS message type: {ws_msg.type}")
        finally:
            self._is_busy = False
            self._current_msg_id = None

    def interrupt(self) -> bool:
        """Send interrupt_request to Colab kernel."""
        if not self._ws or self._connection_state != "connected":
            return False

        interrupt_msg = {
            "header": self._make_header("interrupt_request"),
            "parent_header": {},
            "metadata": {},
            "content": {},
            "channel": "control",
        }
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self._ws.send_json(interrupt_msg))
        except RuntimeError:
            # No running loop - can't send async from sync context
            logger.warning("Cannot send interrupt: no running event loop")
            return False
        return True

    def restart(self) -> bool:
        """Restart Colab runtime by disconnecting and reconnecting.

        Prefer calling _restart_async() directly from async context.
        """
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self._restart_async())
        except RuntimeError:
            logger.warning("Cannot restart Colab kernel: no running event loop")
            return False
        return True

    async def _restart_async(self):
        await self.shutdown_async()
        self._session_id = uuid.uuid4().hex  # Fresh session for new connection
        await self.assign_and_connect()
        self._execution_count = 0

    def shutdown(self):
        """Synchronous shutdown entry point."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.shutdown_async())
            else:
                loop.run_until_complete(self.shutdown_async())
        except RuntimeError:
            pass

    async def shutdown_async(self):
        """Clean shutdown: cancel tasks, close WS, delete session, unassign runtime."""
        # Cancel background tasks
        for task in (self._keep_alive_task, self._token_refresh_task):
            if task:
                task.cancel()
        self._keep_alive_task = None
        self._token_refresh_task = None

        # Close WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        # Close aiohttp session
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._ws_session = None

        # Delete Jupyter session on runtime
        if self._assignment and self._jupyter_session:
            proxy = self._assignment.proxy_info
            await self._api.delete_jupyter_session(
                proxy.url, proxy.token, self._jupyter_session.session_id
            )
        self._jupyter_session = None

        # Unassign runtime
        if self._assignment:
            try:
                await self._api.unassign_kernel(self._assignment.endpoint)
            except Exception as e:
                logger.warning(f"Failed to unassign Colab runtime: {e}")
            self._assignment = None

        self._connection_state = "disconnected"
        logger.info("Colab kernel shutdown complete")

    async def _keep_alive_loop(self):
        """HTTP keep-alive ping every 5 minutes."""
        while True:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            if self._assignment:
                try:
                    await self._api.keep_alive(self._assignment.endpoint)
                    if self._keep_alive_failures:
                        logger.info(
                            "Colab keep-alive recovered after %s failure(s) for %s",
                            self._keep_alive_failures, self._assignment.endpoint,
                        )
                    self._keep_alive_failures = 0
                    if self._connection_state == "degraded" and self._ws and not self._ws.closed:
                        self._connection_state = "connected"
                except Exception as e:
                    self._keep_alive_failures += 1
                    if self._keep_alive_failures >= 2 and self._connection_state == "connected":
                        self._connection_state = "degraded"
                    logger.warning(
                        "Keep-alive failed for %s (failure %s): %s",
                        self._assignment.endpoint, self._keep_alive_failures, e,
                    )

    async def _token_refresh_loop(self):
        """Refresh proxy token before it expires."""
        while True:
            # Sleep until 5 min before expiry
            now = time.time()
            sleep_time = max(60, self._token_expires_at - now - TOKEN_REFRESH_BUFFER)
            await asyncio.sleep(sleep_time)

            if self._assignment:
                try:
                    new_proxy = await self._api.refresh_proxy_token(
                        self._assignment.endpoint
                    )
                    self._assignment.proxy_info.token = new_proxy.token
                    self._token_expires_at = time.time() + new_proxy.token_expires_seconds
                    if self._token_refresh_failures:
                        logger.info(
                            "Colab proxy token refresh recovered after %s failure(s) for %s",
                            self._token_refresh_failures, self._assignment.endpoint,
                        )
                    self._token_refresh_failures = 0
                    if self._connection_state == "degraded" and self._ws and not self._ws.closed:
                        self._connection_state = "connected"
                    logger.info("Colab proxy token refreshed")
                except Exception as e:
                    self._token_refresh_failures += 1
                    if self._token_refresh_failures >= 2 and self._connection_state == "connected":
                        self._connection_state = "degraded"
                    logger.warning(
                        "Proxy token refresh failed for %s (failure %s): %s",
                        self._assignment.endpoint, self._token_refresh_failures, e,
                    )

    @property
    def is_alive(self) -> bool:
        return (
            self._connection_state in {"connected", "degraded"}
            and self._ws is not None
            and not self._ws.closed
        )

    def get_status(self) -> KernelStatus:
        # Show init progress in connection_state when initializing
        conn_state = self._connection_state
        if self._init_status:
            conn_state = f"initializing: {self._init_status}"
        return KernelStatus(
            is_alive=self.is_alive,
            is_busy=self._is_busy,
            execution_count=self._execution_count,
            kernel_type="colab",
            runtime_id=self._assignment.endpoint if self._assignment else None,
            connection_state=conn_state,
        )

    def get_info(self) -> KernelInfo:
        return KernelInfo(
            kernel_type="colab",
            display_name="Google Colab",
            is_remote=True,
            supports_shell_cells=False,
            supports_interrupt=True,
        )

    async def get_namespace_info(self, timeout: float = 5.0) -> dict:
        """Get namespace info by executing introspection code on Colab."""
        if not self.is_alive:
            return {'variables': [], 'functions': []}

        introspection_code = '''
import json as _json, types as _types, inspect as _inspect
_ns_info = {"variables": [], "functions": []}
for _name, _obj in dict(globals()).items():
    if _name.startswith("_") or isinstance(_obj, _types.ModuleType):
        continue
    if callable(_obj) and not isinstance(_obj, type):
        try: _sig = str(_inspect.signature(_obj))
        except: _sig = "(...)"
        _ns_info["functions"].append({"name": _name, "signature": _sig, "type": type(_obj).__name__})
    else:
        _preview = repr(_obj)[:50]
        _ns_info["variables"].append({"name": _name, "type": type(_obj).__name__, "preview": _preview})
print("__NS_INFO__" + _json.dumps(_ns_info))
'''
        result = {'variables': [], 'functions': []}
        async for output in self.execute_streaming(introspection_code):
            if output.output_type == 'stream' and '__NS_INFO__' in str(output.content):
                try:
                    json_str = str(output.content).split('__NS_INFO__', 1)[1].strip()
                    result = json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    pass
        return result


# Register as a kernel backend
def _register_colab_kernel():
    from dialeng.core.registry import registry, KernelRegistration
    registry.register_kernel_type(KernelRegistration(
        name="colab", label="Google Colab", icon="cloud",
        factory=None,  # Colab kernels are created via ColabSessionManager, not a simple factory
        description="Remote Google Colab runtime (requires Google auth)",
        requires_auth=True,
        runtime_options=["cpu", "gpu", "tpu"]
    ))

_register_colab_kernel()
