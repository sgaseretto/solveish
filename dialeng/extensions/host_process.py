"""Isolated extension subprocess implementing Dialeng Extension Host RPC v1."""
from __future__ import annotations

import asyncio
import argparse
import base64
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_VERSION = 1
MAX_RPC_FRAME_BYTES = 4 * 1024 * 1024


def _path_matches(relative: str, pattern: str) -> bool:
    normalized = pattern.strip().replace("\\", "/").strip("/")
    if normalized in {"", ".", "*", "**", "project"}:
        return True
    path = PurePosixPath(relative)
    if any(character in normalized for character in "*?["):
        return path.match(normalized) or path.match(f"{normalized}/**")
    return relative == normalized or relative.startswith(f"{normalized}/")


def _resolve_project_path(identity: dict[str, Any], path: str, access: str) -> Path:
    project = Path(identity["project_root"]).resolve()
    candidate = (project / path).resolve()
    if not candidate.is_relative_to(project):
        raise ValueError("Extension path escapes the project root")
    relative = candidate.relative_to(project).as_posix() or "."
    permission = "filesystem_write" if access == "write" else "filesystem_read"
    patterns = identity.get("permissions", {}).get(permission, [])
    if not isinstance(patterns, list) or not any(_path_matches(relative, str(item)) for item in patterns):
        raise PermissionError(f"Extension manifest does not permit {access} access to {relative!r}")
    return candidate


def _install_audit_policy(identity: dict[str, Any]) -> None:
    """Deny undeclared direct capabilities; supported access goes through the core broker."""

    source_root = Path(identity["source_root"]).resolve()
    project_root = Path(identity["project_root"]).resolve()
    runtime_roots = {
        Path(value).resolve() for value in {sys.prefix, sys.base_prefix, *sys.path}
        if value and Path(value).exists()
    }
    sys.dont_write_bytecode = True

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode = args[1] if len(args) > 1 else "r"
            flags = mode if isinstance(mode, int) else 0
            writing = (
                any(marker in str(mode) for marker in "wax+")
                or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            )
            if not writing and (
                path.is_relative_to(source_root)
                or any(path.is_relative_to(root) for root in runtime_roots)
                or path == Path(os.devnull).resolve()
            ):
                return
            if path.is_relative_to(project_root):
                raise PermissionError(
                    "Direct project filesystem access denied; use the core-side extension broker"
                )
            raise PermissionError(f"Direct filesystem access denied by extension host policy: {path}")
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
            raise PermissionError("Direct network access denied; use the core-side extension broker")
        if (
            event == "subprocess.Popen"
            or event == "os.system"
            or event.startswith("os.spawn")
            or event.startswith("os.posix_spawn")
        ):
            raise PermissionError("Direct process access denied; use the core-side extension broker")

    sys.addaudithook(audit)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _tool_schema(function, name: str) -> dict[str, Any]:
    signature = inspect.signature(function)
    properties = {}
    required = []
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
    for parameter_name, parameter in signature.parameters.items():
        annotation = type_map.get(parameter.annotation, "string")
        properties[parameter_name] = {"type": annotation}
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)
    return {
        "name": name,
        "description": inspect.getdoc(function) or name,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


class BrokeredHttpResponse:
    def __init__(self, payload: dict[str, Any]):
        self.status_code = int(payload.get("status_code", 0))
        self.headers = dict(payload.get("headers") or {})
        self.content = bytes(payload.get("content") or b"")
        self.text = self.content.decode(errors="replace")
        self.truncated = bool(payload.get("truncated"))

    def json(self) -> Any:
        return json.loads(self.content)


class BrokeredHttpClient:
    def __init__(self, context, defaults: dict[str, Any]):
        self.context = context
        self.defaults = defaults

    async def request(self, method: str, url: str, **kwargs) -> BrokeredHttpResponse:
        payload = {**self.defaults, **kwargs}
        result = await self.context.request_http(url, method=method, **payload)
        return BrokeredHttpResponse(result)

    async def get(self, url: str, **kwargs) -> BrokeredHttpResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> BrokeredHttpResponse:
        return await self.request("POST", url, **kwargs)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.aclose()


class HostContext:
    def __init__(self, identity: dict[str, Any], state: dict[str, Any], broker):
        self.__dict__.update(identity)
        self.state = state
        self.logger = logging.LoggerAdapter(
            logging.getLogger(f"dialeng.extension-host.{identity['extension_id']}"),
            {"extension_id": identity["extension_id"]},
        )
        self._cleanups = []
        self._tasks = set()
        self.cancel_signal = asyncio.Event()
        self._broker = broker
        self._settings = dict(identity.get("settings") or {})

    def setting(self, path: str, default: Any = None) -> Any:
        value: Any = self._settings
        for part in str(path).strip(".").split("."):
            if not part:
                continue
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def add_cleanup(self, cleanup):
        self._cleanups.append(cleanup)

    def create_task(self, coro, *, name: str | None = None):
        task = asyncio.create_task(coro, name=f"extension-host:{name or 'task'}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def resolve_project_path(self, path: str, *, access: str = "read", must_exist: bool = False) -> Path:
        candidate = _resolve_project_path(self.__dict__, path, access)
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    async def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        result = await self._broker("filesystem.read_text", {"path": path, "encoding": encoding})
        return str(result["value"])

    async def read_bytes(self, path: str) -> bytes:
        result = await self._broker("filesystem.read_bytes", {"path": path})
        return base64.b64decode(result["value"])

    async def write_text(self, path: str, value: str, *, encoding: str = "utf-8") -> dict[str, Any]:
        return await self._broker(
            "filesystem.write_text", {"path": path, "value": value, "encoding": encoding},
        )

    async def write_bytes(self, path: str, value: bytes) -> dict[str, Any]:
        return await self._broker(
            "filesystem.write_bytes", {"path": path, "value": base64.b64encode(value).decode("ascii")},
        )

    async def request_http(self, url: str, *, method: str = "GET", **kwargs) -> dict[str, Any]:
        result = await self._broker("network.request", {"url": url, "method": method, **kwargs})
        result["content"] = base64.b64decode(result.get("content", ""))
        return result

    def open_http_client(self, **kwargs) -> BrokeredHttpClient:
        if not self.permissions.get("network", False):
            raise PermissionError("Extension manifest does not declare network access")
        return BrokeredHttpClient(self, kwargs)

    async def run_process(self, argv: list[str], *, timeout: float = 30) -> dict[str, Any]:
        return await self._broker("process.run", {"argv": argv, "timeout": timeout})

    async def notify(self, message: str, *, tone: str = "info", notebook_id: str = "") -> None:
        await self._broker("ui.notify", {
            "message": message, "tone": tone, "notebook_id": notebook_id,
        })

    def watch_path(self, path: str, handler, *, name: str = "watch"):
        async def watch():
            previous = await self._broker("filesystem.stat", {"path": path})
            while True:
                await asyncio.sleep(0.5)
                current = await self._broker("filesystem.stat", {"path": path})
                if current == previous:
                    continue
                previous = current
                result = handler({("modified", path)})
                if inspect.isawaitable(result):
                    await result
        return self.create_task(watch(), name=name)

    async def change_cell_type(self, notebook_id: str, cell_id: str, cell_type: str) -> Any:
        result = await self._broker("notebook.change_cell_type", {
            "notebook_id": notebook_id, "cell_id": cell_id, "cell_type": cell_type,
        })
        return result.get("value")

    async def invoke_kernel(self, name: str, **arguments) -> Any:
        result = await self._broker("kernel.invoke", {"name": name, "arguments": arguments})
        return result.get("value")

    async def cleanup(self):
        self.cancel_signal.set()
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for cleanup in reversed(self._cleanups):
            result = cleanup()
            if inspect.isawaitable(result):
                await result


class HostAPI:
    def __init__(self, extension_id: str, context: HostContext):
        self.extension_id = extension_id
        self.context = context
        self.contributions: list[dict[str, Any]] = []
        self.handlers: dict[str, Any] = {}

    def _qualified(self, name: str) -> str:
        if not name or ":" in name:
            raise ValueError("Extension-local names must be non-empty and cannot contain ':'")
        return f"{self.extension_id}:{name}"

    def _handler(self, kind: str, name: str, handler) -> str:
        handler_id = f"{kind}:{name}"
        self.handlers[handler_id] = handler
        return handler_id

    def register_command(self, name: str, *, title: str, description: str, handler, order=50,
                         availability="always", args_schema=None, category="Extensions",
                         keywords=(), default_shortcut=""):
        qualified = self._qualified(name)
        self.contributions.append({
            "kind": "command", "name": qualified, "handler_id": self._handler("command", qualified, handler),
            "title": title, "description": description, "order": order,
            "availability": availability, "args_schema": args_schema or {}, "category": category,
            "keywords": list(keywords), "default_shortcut": default_shortcut,
        })
        return qualified

    def register_action(self, name: str, handler, *, request_schema=None, response_schema=None):
        qualified = self._qualified(name)
        self.contributions.append({
            "kind": "action", "name": qualified, "handler_id": self._handler("action", qualified, handler),
            "request_schema": request_schema or {}, "response_schema": response_schema or {},
        })
        return qualified

    def register_tool(self, name: str, function, *, file_modifying=False):
        provider_name = f"{self.extension_id.replace('.', '-').replace('_', '-')}--{name}"
        self.contributions.append({
            "kind": "tool", "name": provider_name, "handler_id": self._handler("tool", provider_name, function),
            "schema": _tool_schema(function, provider_name), "file_modifying": file_modifying,
        })
        return provider_name

    def register_shortcut(self, name: str, shortcut: str, command: str, *, description=""):
        qualified = self._qualified(name)
        command_name = command if ":" in command else self._qualified(command)
        self.contributions.append({
            "kind": "shortcut", "name": qualified, "shortcut": shortcut,
            "command": command_name, "description": description,
        })
        return qualified

    def register_toolbar_item(self, name: str, renderer=None, *, command="", title="", icon="blocks",
                              position="right", order=50):
        if renderer is not None:
            raise RuntimeError("Python toolbar renderers require privileged in-process mode")
        qualified = self._qualified(name)
        command_name = command if ":" in command else self._qualified(command)
        self.contributions.append({
            "kind": "toolbar", "name": qualified, "command": command_name,
            "title": title or name.replace("-", " ").title(), "icon": icon, "order": order,
        })
        return qualified

    def register_asset(self, name: str, *, kind: str, path: str):
        qualified = self._qualified(name)
        self.contributions.append({"kind": "asset", "name": qualified, "asset_kind": kind, "path": path})
        return qualified

    def register_setting(self, definition):
        self.contributions.append({"kind": "setting", "name": definition.path, "definition": _json_safe(definition)})

    def on(self, event: str, handler, *, name=None, mode="observe", order=50, timeout_seconds=5.0):
        local_name = name or getattr(handler, "__name__", "handler")
        qualified = self._qualified(f"{event}.{local_name}")
        self.contributions.append({
            "kind": "event", "name": qualified, "event": event,
            "handler_id": self._handler("event", qualified, handler), "mode": mode,
            "order": order, "timeout_seconds": timeout_seconds,
        })
        return qualified

    def add_cleanup(self, cleanup):
        self.context.add_cleanup(cleanup)

    def __getattr__(self, name: str):
        if name.startswith("register_"):
            raise RuntimeError(f"{name} requires privileged in-process mode")
        raise AttributeError(name)


class HostRuntime:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.api = None
        self.context = None
        self.operations: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
        self.broker_pending: dict[str, asyncio.Future] = {}
        self.write_lock = asyncio.Lock()
        self.reader = reader
        self.writer = writer

    async def send(self, payload: dict[str, Any]):
        encoded = (json.dumps(
            _json_safe({"version": PROTOCOL_VERSION, **payload}), separators=(",", ":"),
        ) + "\n").encode()
        if len(encoded) > MAX_RPC_FRAME_BYTES:
            raise RuntimeError("Extension host RPC frame exceeds the 4 MiB limit")
        async with self.write_lock:
            self.writer.write(encoded)
            await self.writer.drain()

    async def broker_request(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        request_id = f"broker-{uuid.uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self.broker_pending[request_id] = future
        await self.send({
            "id": request_id,
            "method": "broker",
            "params": {"operation": operation, "arguments": arguments},
        })
        try:
            return await asyncio.wait_for(future, timeout=120)
        finally:
            self.broker_pending.pop(request_id, None)

    async def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        root = Path(params["root"]).resolve()
        _install_audit_policy(params["identity"])
        entrypoint = params["entrypoint"]
        module_name, function_name = entrypoint.split(":", 1)
        module_path = root / (module_name.replace(".", "/") + ".py")
        package_path = root / module_name.replace(".", "/") / "__init__.py"
        if module_path.is_file() or package_path.is_file():
            path = module_path if module_path.is_file() else package_path
            unique_name = f"_dialeng_host_{params['identity']['extension_id'].replace('.', '_')}"
            spec = importlib.util.spec_from_file_location(unique_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not import {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        setup = getattr(module, function_name, None)
        if not callable(setup):
            raise ImportError(f"Entrypoint function not found: {function_name}")
        self.context = HostContext(
            params["identity"], dict(params.get("state") or {}), self.broker_request,
        )
        self.api = HostAPI(params["identity"]["extension_id"], self.context)
        result = setup(self.api, self.context)
        if inspect.isawaitable(result):
            await result
        return {"contributions": self.api.contributions, "state": self.context.state}

    async def invoke(self, params: dict[str, Any]) -> Any:
        if self.api is None:
            raise RuntimeError("Extension host is not initialized")
        handler = self.api.handlers[params["handler_id"]]
        self.context._settings = dict(params.get("settings") or {})
        kind = params["kind"]
        arguments = dict(params.get("arguments") or {})
        operation_id = str(params.get("operation_id") or "")
        cancel_event = asyncio.Event()

        async def report_progress(message: str, *, current=None, total=None):
            await self.send({"method": "progress", "params": {
                "operation_id": operation_id, "message": message, "current": current, "total": total,
            }})

        if kind == "command":
            command_context = {
                **arguments,
                "args": dict(arguments.get("args") or {}),
                "cancel_event": cancel_event,
                "report_progress": report_progress,
            }
            result = handler(command_context)
        elif kind == "action":
            result = handler(**arguments)
        elif kind == "tool":
            result = handler(**arguments)
        elif kind == "event":
            result = handler(arguments.get("payload") or {})
        else:
            raise ValueError(f"Unknown invocation kind: {kind}")
        if inspect.isawaitable(result):
            result = await result
        return {"value": result, "state": self.context.state}

    async def dispatch(self, request: dict[str, Any]):
        request_id = str(request.get("id") or "")
        try:
            if request.get("version") != PROTOCOL_VERSION:
                raise ValueError("Unsupported extension host protocol version")
            method = request.get("method")
            if method == "initialize":
                result = await self.initialize(request.get("params") or {})
            elif method == "invoke":
                result = await self.invoke(request.get("params") or {})
            elif method == "heartbeat":
                result = {"alive": True}
            elif method == "cancel":
                operation_id = str((request.get("params") or {}).get("operation_id") or "")
                active = self.operations.get(operation_id)
                if active:
                    active[1].set()
                    active[0].cancel()
                result = {"cancelled": bool(active)}
            elif method == "shutdown":
                if self.context:
                    await self.context.cleanup()
                result = {"stopped": True, "state": self.context.state if self.context else {}}
            else:
                raise ValueError(f"Unknown RPC method: {method}")
            await self.send({"id": request_id, "ok": True, "result": result})
        except asyncio.CancelledError:
            await self.send({"id": request_id, "ok": False, "error": {"type": "CancelledError", "message": "Operation cancelled"}})
        except BaseException as exc:
            await self.send({
                "id": request_id, "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)},
            })

    async def run(self):
        while True:
            line = await self.reader.readline()
            if not line:
                break
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                await self.send({"ok": False, "error": {"type": "ProtocolError", "message": "Malformed JSON"}})
                continue
            request_id = str(request.get("id") or "")
            if "method" not in request and request_id in self.broker_pending:
                future = self.broker_pending.get(request_id)
                if future and not future.done():
                    if request.get("ok"):
                        future.set_result(request.get("result") or {})
                    else:
                        error = request.get("error") or {}
                        future.set_exception(RuntimeError(
                            f"{error.get('type', 'BrokerError')}: {error.get('message', 'Broker request failed')}"
                        ))
                continue
            method = request.get("method")
            operation_id = str((request.get("params") or {}).get("operation_id") or "")
            task = asyncio.create_task(self.dispatch(request))
            if method == "invoke" and operation_id:
                self.operations[operation_id] = (task, asyncio.Event())
                task.add_done_callback(lambda _task, key=operation_id: self.operations.pop(key, None))
            if method == "shutdown":
                await task
                break


async def _connect(endpoint: dict[str, Any]) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if endpoint.get("transport") == "unix":
        return await asyncio.open_unix_connection(
            endpoint["path"], limit=MAX_RPC_FRAME_BYTES,
        )
    if endpoint.get("transport") == "tcp":
        return await asyncio.open_connection(
            endpoint.get("host", "127.0.0.1"), int(endpoint["port"]), limit=MAX_RPC_FRAME_BYTES,
        )
    raise ValueError("Unsupported extension host RPC transport")


async def _main(endpoint: dict[str, Any], token: str) -> None:
    reader, writer = await _connect(endpoint)
    hello = json.dumps({
        "version": PROTOCOL_VERSION, "type": "hello", "token": token,
    }, separators=(",", ":")) + "\n"
    writer.write(hello.encode())
    await writer.drain()
    try:
        await HostRuntime(reader, writer).run()
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1)
        except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError):
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", required=True)
    arguments = parser.parse_args()
    asyncio.run(_main(json.loads(arguments.endpoint), arguments.token))
