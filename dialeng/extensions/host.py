"""Supervisor and proxy contributions for isolated extension hosts."""
from __future__ import annotations

import asyncio
import ast
import base64
import json
import logging
import os
import secrets
import signal
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dialeng.services.settings_service import SettingDefinition

from .context import ExtensionContribution
from .events import EventHandlerRegistration
from .manifest import ExtensionSource
from .permissions import resolve_permitted_project_path
from .resources import (
    ActionRegistration,
    AssetRegistration,
    CommandRegistration,
    ShortcutRegistration,
    ToolRegistration,
    ToolbarRegistration,
)


logger = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
MAX_RPC_FRAME_BYTES = 4 * 1024 * 1024
PRIVILEGED_CONTRIBUTIONS = {
    "callbacks", "cell_types", "kernels", "panels", "providers", "services", "status",
}
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "COOKIE", "AUTHORIZATION", "API_KEY", "AWS_")


class ExtensionHostError(RuntimeError):
    pass


def source_runtime_mode(source: ExtensionSource) -> str:
    privileged = sorted(
        name for name in PRIVILEGED_CONTRIBUTIONS if source.manifest.contributes.get(name)
    )
    if privileged:
        return "privileged_in_process"
    try:
        module_name = source.manifest.entrypoint.split(":", 1)[0]
        module_path = source.root / (module_name.replace(".", "/") + ".py")
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        privileged_methods = {
            "register_callback", "register_cell_type", "register_kernel", "register_llm_converter",
            "register_panel", "register_provider", "register_renderer", "register_service",
            "register_settings_section", "register_status_item",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.startswith("dialeng.core") or node.module.startswith("fasthtml")
            ):
                return "privileged_in_process"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in privileged_methods:
                    return "privileged_in_process"
    except (OSError, SyntaxError):
        pass
    return "isolated"


def _sanitized_environment() -> dict[str, str]:
    allowed = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONIOENCODING"}
    environment = {
        key: value for key, value in os.environ.items()
        if key in allowed and not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _resource_limits():
    if os.name != "posix":
        return None

    def apply():
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        if sys.platform.startswith("linux"):
            limit = 768 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


@dataclass
class ExtensionHost:
    source: ExtensionSource
    identity: dict[str, Any]
    initial_state: dict[str, Any]
    persist_state: Any
    host_script: Path | None = None
    broker_services: dict[str, Any] = field(default_factory=dict)
    process: asyncio.subprocess.Process | None = None
    contributions: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    progress_handlers: dict[str, Any] = field(default_factory=dict)
    reader_task: asyncio.Task | None = None
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    monitor_task: asyncio.Task | None = None
    rpc_server: asyncio.AbstractServer | None = None
    rpc_reader: asyncio.StreamReader | None = None
    rpc_writer: asyncio.StreamWriter | None = None
    rpc_connected: asyncio.Future | None = None
    rpc_token: str = ""
    rpc_endpoint: dict[str, Any] = field(default_factory=dict)
    socket_directory: Path | None = None
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    restart_count: int = 0
    consecutive_failures: int = 0
    disabled: bool = False
    last_heartbeat: float = 0.0
    memory_limit_bytes: int = 768 * 1024 * 1024
    monitor_interval: float = 5.0
    restart_backoff_base: float = 0.2
    restart_backoff_max: float = 2.0

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process and self.process.returncode is None else None

    @property
    def alive(self) -> bool:
        return (
            self.pid is not None
            and self.rpc_writer is not None
            and not self.rpc_writer.is_closing()
            and not self.disabled
        )

    async def start(self) -> list[dict[str, Any]]:
        if self.disabled:
            raise ExtensionHostError(f"Extension host {self.source.manifest.id} is disabled after repeated failures")
        if self.process is not None or self.rpc_writer is not None or self.rpc_server is not None:
            await self.terminate()
        try:
            self.identity["settings"] = self._settings_snapshot()
            await self._open_rpc_server()
            script = self.host_script or Path(__file__).with_name("host_process.py")
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-I", str(script),
                f"--endpoint={json.dumps(self.rpc_endpoint, separators=(',', ':'))}",
                f"--token={self.rpc_token}",
                cwd=self.source.root,
                env=_sanitized_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_RPC_FRAME_BYTES,
                start_new_session=True,
                preexec_fn=_resource_limits(),
            )
            self.stdout_task = asyncio.create_task(
                self._read_process_stream(self.process.stdout, "stdout"),
                name=f"extension-host-stdout:{self.source.manifest.id}",
            )
            self.stderr_task = asyncio.create_task(
                self._read_process_stream(self.process.stderr, "stderr"),
                name=f"extension-host-stderr:{self.source.manifest.id}",
            )
            assert self.rpc_connected is not None
            await asyncio.wait_for(asyncio.shield(self.rpc_connected), timeout=8)
            self.reader_task = asyncio.create_task(
                self._read_rpc(), name=f"extension-host-reader:{self.source.manifest.id}",
            )
            result = await self.request("initialize", {
                "root": str(self.source.root),
                "entrypoint": self.source.manifest.entrypoint,
                "identity": self.identity,
                "state": self.initial_state,
            }, timeout=12)
        except Exception:
            await self.terminate()
            raise
        self.contributions = list(result.get("contributions") or [])
        self.initial_state = dict(result.get("state") or {})
        self.persist_state(self.initial_state)
        self.last_heartbeat = time.monotonic()
        self.monitor_task = asyncio.create_task(
            self._monitor(), name=f"extension-host-monitor:{self.source.manifest.id}",
        )
        return self.contributions

    def _settings_snapshot(self) -> dict[str, Any]:
        from dialeng.services.dialeng_config import get_config

        value = get_config().raw_config.get("extensions", {}).get(self.source.manifest.id, {})
        return dict(value) if isinstance(value, dict) else {}

    async def _open_rpc_server(self) -> None:
        loop = asyncio.get_running_loop()
        self.rpc_token = secrets.token_urlsafe(32)
        self.rpc_connected = loop.create_future()
        if os.name == "posix" and hasattr(asyncio, "start_unix_server"):
            self.socket_directory = Path(tempfile.mkdtemp(prefix="dialeng-extension-host-"))
            socket_path = self.socket_directory / "rpc.sock"
            self.rpc_server = await asyncio.start_unix_server(
                self._accept_rpc, path=socket_path, limit=MAX_RPC_FRAME_BYTES,
            )
            self.rpc_endpoint = {"transport": "unix", "path": str(socket_path)}
            return
        self.rpc_server = await asyncio.start_server(
            self._accept_rpc, host="127.0.0.1", port=0, limit=MAX_RPC_FRAME_BYTES,
        )
        address = self.rpc_server.sockets[0].getsockname()
        self.rpc_endpoint = {"transport": "tcp", "host": "127.0.0.1", "port": int(address[1])}

    async def _accept_rpc(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            payload = json.loads(line)
            if (
                payload.get("version") != PROTOCOL_VERSION
                or payload.get("type") != "hello"
                or not secrets.compare_digest(str(payload.get("token") or ""), self.rpc_token)
            ):
                raise ExtensionHostError("Extension host authentication failed")
            if self.rpc_writer is not None:
                raise ExtensionHostError("Extension host RPC already connected")
            self.rpc_reader = reader
            self.rpc_writer = writer
            if self.rpc_connected and not self.rpc_connected.done():
                self.rpc_connected.set_result(True)
        except Exception as exc:
            writer.close()
            try:
                await writer.wait_closed()
            except (asyncio.CancelledError, ConnectionError, RuntimeError):
                pass
            if self.rpc_connected and not self.rpc_connected.done():
                self.rpc_connected.set_exception(exc)

    async def _read_rpc(self):
        assert self.rpc_reader
        try:
            while line := await self.rpc_reader.readline():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    await self._fail_all(ExtensionHostError(f"Malformed extension host RPC: {exc}"))
                    await self.terminate()
                    return
                if payload.get("version") != PROTOCOL_VERSION:
                    await self._fail_all(ExtensionHostError("Extension host protocol version mismatch"))
                    await self.terminate()
                    return
                if payload.get("method") == "broker":
                    asyncio.create_task(self._handle_broker(payload))
                    continue
                if payload.get("method") == "progress":
                    params = payload.get("params") or {}
                    callback = self.progress_handlers.get(str(params.get("operation_id") or ""))
                    if callback:
                        result = callback(
                            str(params.get("message") or ""),
                            current=params.get("current"), total=params.get("total"),
                        )
                        if asyncio.iscoroutine(result):
                            await result
                    continue
                request_id = str(payload.get("id") or "")
                future = self.pending.pop(request_id, None)
                if future is None or future.done():
                    continue
                if payload.get("ok"):
                    future.set_result(payload.get("result") or {})
                else:
                    error = payload.get("error") or {}
                    future.set_exception(ExtensionHostError(
                        f"{error.get('type', 'ExtensionHostError')}: {error.get('message', 'Host request failed')}"
                    ))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer = self.rpc_writer
            self.rpc_writer = None
            self.rpc_reader = None
            if writer is not None:
                writer.close()
            if self.pending:
                self.consecutive_failures += 1
                await self._fail_all(ExtensionHostError(
                    f"Extension host exited unexpectedly ({self.process.returncode if self.process else 'unknown'})"
                ))

    async def _read_process_stream(self, stream: asyncio.StreamReader | None, channel: str):
        if stream is None:
            return
        while line := await stream.readline():
            log = logger.warning if channel == "stderr" else logger.info
            log(
                "Extension host %s (extension=%s): %s", channel,
                self.source.manifest.id, line.decode(errors="replace").rstrip()[:1000],
            )

    async def _send_rpc(self, payload: dict[str, Any]) -> None:
        if self.rpc_writer is None or self.rpc_writer.is_closing():
            raise ExtensionHostError("Extension host RPC is not connected")
        encoded = (json.dumps({"version": PROTOCOL_VERSION, **payload}, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_RPC_FRAME_BYTES:
            raise ExtensionHostError("Extension host RPC frame exceeds the 4 MiB limit")
        async with self.write_lock:
            self.rpc_writer.write(encoded)
            await self.rpc_writer.drain()

    async def _handle_broker(self, request: dict[str, Any]) -> None:
        request_id = str(request.get("id") or "")
        params = request.get("params") or {}
        try:
            result = await self._broker_request(str(params.get("operation") or ""), params.get("arguments") or {})
            await self._send_rpc({"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            await self._send_rpc({
                "id": request_id,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })

    async def _broker_request(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        permissions = self.source.manifest.permissions
        project_root = Path(self.identity["project_root"])
        if operation in {"filesystem.read_text", "filesystem.read_bytes", "filesystem.stat"}:
            path = resolve_permitted_project_path(
                project_root, arguments.get("path", ""), permissions, access="read", must_exist=True,
            )
            if operation.endswith("stat"):
                stat = await asyncio.to_thread(path.stat)
                return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "is_dir": path.is_dir()}
            if operation.endswith("read_text"):
                return {"value": await asyncio.to_thread(path.read_text, encoding=arguments.get("encoding", "utf-8"))}
            return {"value": base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode("ascii")}
        if operation in {"filesystem.write_text", "filesystem.write_bytes"}:
            path = resolve_permitted_project_path(
                project_root, arguments.get("path", ""), permissions, access="write",
            )
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            if operation.endswith("write_text"):
                value = str(arguments.get("value", ""))
                await asyncio.to_thread(path.write_text, value, encoding=arguments.get("encoding", "utf-8"))
            else:
                value = base64.b64decode(str(arguments.get("value", "")), validate=True)
                await asyncio.to_thread(path.write_bytes, value)
            return {"path": path.relative_to(project_root.resolve()).as_posix(), "bytes": len(value)}
        if operation == "network.request":
            if not permissions.get("network", False):
                raise PermissionError("Extension manifest does not declare network access")
            import httpx

            timeout = min(max(float(arguments.get("timeout", 10)), 0.1), 30)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                response = await client.request(
                    str(arguments.get("method", "GET")).upper(),
                    str(arguments.get("url", "")),
                    headers=dict(arguments.get("headers") or {}),
                    params=arguments.get("params"),
                    json=arguments.get("json"),
                    content=arguments.get("content"),
                )
            body = response.content[:1_000_000]
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": base64.b64encode(body).decode("ascii"),
                "truncated": len(response.content) > len(body),
            }
        if operation == "process.run":
            if not permissions.get("subprocess", False):
                raise PermissionError("Extension manifest does not declare subprocess access")
            argv = arguments.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
                raise ValueError("process.run requires a non-empty string argv list")
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=project_root,
                env=_sanitized_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=min(max(float(arguments.get("timeout", 30)), 0.1), 120),
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ExtensionHostError("Brokered subprocess timed out")
            return {
                "returncode": process.returncode,
                "stdout": stdout[:1_000_000].decode(errors="replace"),
                "stderr": stderr[:1_000_000].decode(errors="replace"),
                "truncated": len(stdout) > 1_000_000 or len(stderr) > 1_000_000,
            }
        if operation == "notebook.change_cell_type":
            if not permissions.get("notebook", False):
                raise PermissionError("Extension manifest does not declare notebook access")
            handler = self.broker_services.get("change_cell_type")
            if handler is None:
                raise RuntimeError("Notebook mutation service is unavailable")
            result = handler(
                str(arguments.get("notebook_id", "")),
                str(arguments.get("cell_id", "")),
                str(arguments.get("cell_type", "")),
            )
            if asyncio.iscoroutine(result):
                result = await result
            return {"value": result}
        if operation == "kernel.invoke":
            if not permissions.get("kernel", False):
                raise PermissionError("Extension manifest does not declare kernel access")
            handler = self.broker_services.get("kernel_operation")
            if handler is None:
                raise RuntimeError("Kernel broker service is unavailable")
            result = handler(str(arguments.get("name", "")), dict(arguments.get("arguments") or {}))
            if asyncio.iscoroutine(result):
                result = await result
            return {"value": result}
        if operation == "setting.get":
            from dialeng.services.dialeng_config import get_config

            value: Any = get_config().raw_config
            full_path = f"extensions.{self.source.manifest.id}.{str(arguments.get('path', '')).strip('.')}"
            for part in full_path.split("."):
                if not isinstance(value, dict) or part not in value:
                    return {"value": arguments.get("default")}
                value = value[part]
            return {"value": value}
        if operation == "ui.notify":
            handler = self.broker_services.get("notify")
            if handler is None:
                logger.info(
                    "Extension notification (extension=%s): %s",
                    self.source.manifest.id, str(arguments.get("message", "")),
                )
                return {"value": None}
            result = handler(
                str(arguments.get("message", "")),
                tone=str(arguments.get("tone", "info")),
                notebook_id=str(arguments.get("notebook_id", "")),
            )
            if asyncio.iscoroutine(result):
                result = await result
            return {"value": result}
        raise ValueError(f"Unknown extension broker operation: {operation}")

    async def _monitor(self):
        try:
            while self.process and self.process.returncode is None:
                await asyncio.sleep(self.monitor_interval)
                try:
                    await self.request("heartbeat", {}, timeout=2)
                    self.last_heartbeat = time.monotonic()
                    rss = await asyncio.to_thread(self._rss_bytes)
                    if rss and rss > self.memory_limit_bytes:
                        raise ExtensionHostError(f"Extension host exceeded memory limit ({rss} bytes)")
                except Exception as exc:
                    self.consecutive_failures += 1
                    logger.error("Extension host health failure (extension=%s): %s", self.source.manifest.id, exc)
                    await self.terminate()
                    return
        except asyncio.CancelledError:
            return

    def _rss_bytes(self) -> int:
        if not self.pid:
            return 0
        import subprocess

        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.pid)], text=True, capture_output=True, timeout=2,
        )
        try:
            return int(completed.stdout.strip() or "0") * 1024
        except ValueError:
            return 0

    async def _fail_all(self, error: Exception):
        for future in self.pending.values():
            if not future.done():
                future.set_exception(error)
        self.pending.clear()

    async def ensure_running(self):
        if self.alive:
            return
        if self.consecutive_failures >= 3:
            self.disabled = True
            raise ExtensionHostError(f"Extension host {self.source.manifest.id} disabled after repeated failures")
        self.restart_count += 1
        await asyncio.sleep(min(
            self.restart_backoff_base * (2 ** (self.restart_count - 1)),
            self.restart_backoff_max,
        ))
        await self.start()

    async def request(self, method: str, params: dict[str, Any], *, timeout: float = 30) -> dict[str, Any]:
        if method != "initialize":
            await self.ensure_running()
        if not self.process or not self.rpc_writer:
            raise ExtensionHostError("Extension host is not running")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        await self._send_rpc({"id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self.pending.pop(request_id, None)
            self.consecutive_failures += 1
            await self.terminate()
            raise ExtensionHostError(f"Extension host request timed out: {method}") from exc

    async def invoke(
        self,
        kind: str,
        handler_id: str,
        arguments: dict[str, Any],
        *,
        operation_id: str = "",
        progress=None,
        timeout: float = 30,
    ) -> Any:
        await self.ensure_running()
        operation_id = operation_id or f"host-operation-{uuid.uuid4().hex[:12]}"
        if progress:
            self.progress_handlers[operation_id] = progress
        try:
            result = await self.request("invoke", {
                "kind": kind,
                "handler_id": handler_id,
                "arguments": arguments,
                "operation_id": operation_id,
                "settings": self._settings_snapshot(),
            }, timeout=timeout)
            state = result.get("state")
            if isinstance(state, dict):
                self.initial_state = state
                self.persist_state(state)
            self.consecutive_failures = 0
            return result.get("value")
        except asyncio.CancelledError:
            try:
                await self.request("cancel", {"operation_id": operation_id}, timeout=2)
            except Exception:
                await self.terminate()
            raise
        finally:
            self.progress_handlers.pop(operation_id, None)

    async def stop(self):
        if self.process and self.process.returncode is None:
            try:
                result = await self.request("shutdown", {}, timeout=3)
                state = result.get("state")
                if isinstance(state, dict):
                    self.initial_state = state
                    self.persist_state(state)
            except Exception:
                pass
        await self.terminate()

    async def terminate(self):
        current = asyncio.current_task()
        for task in (self.monitor_task, self.reader_task, self.stdout_task, self.stderr_task):
            if task and task is not current and not task.done():
                task.cancel()
        await self._fail_all(ExtensionHostError("Extension host terminated"))
        if self.rpc_writer is not None:
            self.rpc_writer.close()
            try:
                await asyncio.wait_for(self.rpc_writer.wait_closed(), timeout=1)
            except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError, RuntimeError):
                pass
        self.rpc_writer = None
        self.rpc_reader = None
        if self.rpc_server is not None:
            self.rpc_server.close()
            await self.rpc_server.wait_closed()
        self.rpc_server = None
        process = self.process
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                await process.wait()
        self.process = None
        background = [
            task for task in (self.reader_task, self.stdout_task, self.stderr_task)
            if task and task is not current
        ]
        if background:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background, return_exceptions=True), timeout=1,
                )
            except asyncio.TimeoutError:
                pass
        self.reader_task = None
        self.stdout_task = None
        self.stderr_task = None
        self.monitor_task = None
        if self.socket_directory is not None:
            await asyncio.to_thread(shutil.rmtree, self.socket_directory, True)
        self.socket_directory = None
        self.rpc_endpoint = {}
        self.rpc_token = ""

    def health(self) -> dict[str, Any]:
        return {
            "mode": "isolated",
            "pid": self.pid,
            "alive": self.alive,
            "restart_count": self.restart_count,
            "consecutive_failures": self.consecutive_failures,
            "disabled": self.disabled,
            "last_heartbeat": self.last_heartbeat,
            "pending_operations": len(self.pending),
            "rpc_transport": self.rpc_endpoint.get("transport", ""),
            "rpc_protocol": PROTOCOL_VERSION,
        }


def proxy_contributions(host: ExtensionHost) -> list[ExtensionContribution]:
    """Convert host declarations into core-owned proxy registrations."""
    result = []
    extension_id = host.source.manifest.id
    version = host.source.manifest.version

    def contribution(kind: str, name: str, value: Any):
        result.append(ExtensionContribution(kind, name, value, owner=extension_id, extension_version=version))

    for item in host.contributions:
        kind = item["kind"]
        name = item["name"]
        handler_id = item.get("handler_id", "")
        if kind == "command":
            async def command_handler(context, _handler=handler_id):
                return await host.invoke(
                    "command", _handler,
                    {key: value for key, value in context.items() if key not in {"cancel_event", "report_progress"}},
                    operation_id=context.get("operation_id", ""), progress=context.get("report_progress"),
                )
            value = CommandRegistration(
                name, item["title"], item["description"], command_handler,
                item.get("order", 50), item.get("availability", "always"), item.get("args_schema") or {},
                item.get("category", "Extensions"), tuple(item.get("keywords") or ()),
                item.get("default_shortcut", ""), "",
            )
            contribution("extension.command", name, value)
        elif kind == "action":
            async def action_handler(_handler=handler_id, **kwargs):
                return await host.invoke("action", _handler, kwargs)
            contribution("extension.action", name, ActionRegistration(
                name, action_handler, item.get("request_schema") or {}, item.get("response_schema") or {},
            ))
        elif kind == "tool":
            async def tool_handler(_handler=handler_id, **kwargs):
                return await host.invoke("tool", _handler, kwargs)
            contribution("extension.tool", name, ToolRegistration(
                name, tool_handler, item.get("schema") or {}, bool(item.get("file_modifying")),
            ))
        elif kind == "event":
            event_timeout = float(item.get("timeout_seconds", 5))
            async def event_handler(payload, _handler=handler_id, _timeout=event_timeout):
                return await host.invoke("event", _handler, {"payload": payload}, timeout=_timeout)
            contribution("extension.event", name, EventHandlerRegistration(
                name, item["event"], event_handler, item.get("mode", "observe"),
                int(item.get("order", 50)), float(item.get("timeout_seconds", 5)),
            ))
        elif kind == "shortcut":
            contribution("extension.shortcut", name, ShortcutRegistration(
                name, item["shortcut"], item["command"], item.get("description", ""),
            ))
        elif kind == "toolbar":
            contribution("extension.toolbar", name, ToolbarRegistration(
                name, item["title"], item["command"], item.get("icon", "blocks"), int(item.get("order", 50)),
            ))
        elif kind == "asset":
            contribution("extension.asset", name, AssetRegistration(name, item["asset_kind"], item["path"]))
        elif kind == "setting":
            definition = dict(item["definition"])
            local_path = str(definition.pop("path")).strip(".")
            definition["path"] = f"extensions.{extension_id}.{local_path}"
            contribution("settings.definition", definition["path"], SettingDefinition(**definition))
        else:
            raise ExtensionHostError(f"Unsupported isolated contribution: {kind}")
    return result
