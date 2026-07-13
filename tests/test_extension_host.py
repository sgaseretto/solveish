"""Adversarial tests for per-extension process isolation and supervision."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from dialeng.extensions.api import ExtensionContext
from dialeng.extensions.host import ExtensionHost, ExtensionHostError
from dialeng.extensions.manager import ExtensionManager
from dialeng.extensions.resources import commands
from dialeng.extensions.state import ExtensionStateStore
from dialeng.extensions.trust import ExtensionTrustStore


def _write_extension(
    root: Path,
    extension_id: str,
    setup_source: str,
    *,
    contributes: str = "commands = true",
    permissions: str = """filesystem_read = []
filesystem_write = []
network = false
subprocess = false
browser_assets = false""",
):
    root.mkdir(parents=True)
    (root / "extension.toml").write_text(
        f'''schema_version = 1
id = "{extension_id}"
name = "{extension_id}"
version = "1.0.0"
description = "Isolated host fixture"
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "project"

[permissions]
{permissions}

[contributes]
{contributes}
''',
        encoding="utf-8",
    )
    (root / "extension.py").write_text(setup_source, encoding="utf-8")


def _manager(project: Path, tmp_path: Path, *, context_services=None) -> ExtensionManager:
    return ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "no-user",
        context_services=context_services,
    )


def test_rpc_safe_extension_runs_out_of_process_and_persists_state(tmp_path):
    project = tmp_path / "project"
    root = project / ".dialeng" / "extensions" / "stateful"
    _write_extension(root, "test.host-state", '''def setup(api, ctx):
    async def increment(command_context):
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        return {"count": ctx.state["count"], "pid": __import__("os").getpid()}
    api.register_command("increment", title="Increment", description="Increment", handler=increment)
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-state", require_trust=False)
        assert active.runtime_mode == "isolated"
        assert active.host.pid and active.host.pid != __import__("os").getpid()
        result = await commands["test.host-state:increment"].handler({
            "operation_id": "host-state-one", "notebook_id": "demo", "project_root": str(project),
            "args": {}, "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
        })
        assert result["count"] == 1
        assert result["pid"] == active.host.pid
        assert manager.state_store.runtime_state(project, "test.host-state")["count"] == 1
        await manager.shutdown()
        assert active.host.alive is False

    asyncio.run(scenario())


def test_host_crash_does_not_stop_unrelated_extension_and_restarts_lazily(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/crash", "test.host-crash", '''def setup(api, ctx):
    def crash(command_context):
        __import__("os")._exit(37)
    api.register_command("crash", title="Crash", description="Crash", handler=crash)
    api.register_command("healthy", title="Healthy", description="Healthy", handler=lambda context: {"ok": True})
''')
    _write_extension(project / ".dialeng/extensions/other", "test.host-other", '''def setup(api, ctx):
    api.register_command("ping", title="Ping", description="Ping", handler=lambda context: {"pong": True})
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        crashed = await manager.activate("test.host-crash", require_trust=False)
        other = await manager.activate("test.host-other", require_trust=False)
        with pytest.raises(ExtensionHostError, match="exited unexpectedly"):
            await commands["test.host-crash:crash"].handler({
                "operation_id": "host-crash", "args": {}, "notebook_id": "demo",
                "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
            })
        assert other.host.alive
        pong = await commands["test.host-other:ping"].handler({
            "operation_id": "other-ping", "args": {}, "notebook_id": "demo",
            "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
        })
        assert pong == {"pong": True}
        healthy = await commands["test.host-crash:healthy"].handler({
            "operation_id": "host-restart", "args": {}, "notebook_id": "demo",
            "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
        })
        assert healthy == {"ok": True}
        assert crashed.host.restart_count == 1
        await manager.shutdown()

    asyncio.run(scenario())


def test_repeated_crash_restart_cycles_leave_one_host_and_no_socket_leaks(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/restart", "test.host-restart", '''def setup(api, ctx):
    api.register_command("crash", title="Crash", description="Crash", handler=lambda context: __import__("os")._exit(45))
    api.register_command("healthy", title="Healthy", description="Healthy", handler=lambda context: {"ok": True})
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-restart", require_trust=False)
        active.host.restart_backoff_base = 0.001
        active.host.restart_backoff_max = 0.01
        handlers = {
            item["name"].rsplit(":", 1)[-1]: item["handler_id"]
            for item in active.host.contributions
        }
        retired_sockets = []
        for _index in range(25):
            retired_sockets.append(active.host.socket_directory)
            with pytest.raises(ExtensionHostError, match="exited unexpectedly"):
                await active.host.invoke("command", handlers["crash"], {"args": {}})
            assert await active.host.invoke("command", handlers["healthy"], {"args": {}}) == {"ok": True}
            assert active.host.alive
            assert not any(path and path.exists() for path in retired_sockets)
        assert active.host.restart_count == 25
        final_socket = active.host.socket_directory
        await manager.shutdown()
        assert final_socket and not final_socket.exists()
        assert not active.host.alive

    asyncio.run(scenario())


def test_blocking_host_is_terminated_and_extension_stdout_cannot_corrupt_rpc(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/bad", "test.host-bad", '''def setup(api, ctx):
    def block(command_context):
        while True: pass
    def malformed(command_context):
        print("not-json", flush=True)
        return {"ok": True}
    api.register_command("block", title="Block", description="Block", handler=block)
    api.register_command("malformed", title="Malformed", description="Malformed", handler=malformed)
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-bad", require_trust=False)
        block_id = next(item["handler_id"] for item in active.host.contributions if item["name"].endswith(":block"))
        started = time.monotonic()
        operation = asyncio.create_task(active.host.invoke("command", block_id, {"args": {}}, timeout=0.25))
        loop_delays = []
        previous = time.monotonic()
        while not operation.done():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            loop_delays.append(now - previous)
            previous = now
        with pytest.raises(ExtensionHostError, match="timed out"):
            await operation
        assert time.monotonic() - started < 3
        assert loop_delays and max(loop_delays) < 0.1
        malformed_id = next(item["handler_id"] for item in active.host.contributions if item["name"].endswith(":malformed"))
        result = await active.host.invoke("command", malformed_id, {"args": {}}, timeout=3)
        assert result == {"ok": True}
        await manager.shutdown()

    asyncio.run(scenario())


def test_host_kill_during_tool_and_event_calls_returns_terminal_errors(tmp_path):
    project = tmp_path / "project"
    _write_extension(
        project / ".dialeng/extensions/invocations", "test.host-invocations",
        '''def setup(api, ctx):
    def crash_tool(value: str) -> str:
        """Crash during a tool call."""
        __import__("os")._exit(41)
    def crash_event(payload):
        __import__("os")._exit(42)
    api.register_tool("crash_tool", crash_tool)
    api.on("notebook.opened", crash_event)
''', contributes="tools = true\nevents = true",
    )
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-invocations", require_trust=False)
        tool = next(item for item in active.host.contributions if item["kind"] == "tool")
        event = next(item for item in active.host.contributions if item["kind"] == "event")
        with pytest.raises(ExtensionHostError, match="exited unexpectedly"):
            await active.host.invoke("tool", tool["handler_id"], {"value": "x"})
        with pytest.raises(ExtensionHostError, match="exited unexpectedly"):
            await active.host.invoke("event", event["handler_id"], {"payload": {"id": "demo"}})
        assert active.host.restart_count == 1
        await manager.shutdown()

    asyncio.run(scenario())


def test_malformed_private_socket_frame_terminates_host_with_protocol_error(tmp_path):
    project = tmp_path / "project"
    _write_extension(
        project / ".dialeng/extensions/protocol", "test.host-protocol",
        'def setup(api, ctx): pass\n', contributes="commands = false",
    )
    fake_host = tmp_path / "fake_host.py"
    fake_host.write_text('''import argparse, asyncio, json
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    endpoint = json.loads(args.endpoint)
    if endpoint["transport"] == "unix":
        reader, writer = await asyncio.open_unix_connection(endpoint["path"])
    else:
        reader, writer = await asyncio.open_connection(endpoint["host"], endpoint["port"])
    writer.write((json.dumps({"version": 1, "type": "hello", "token": args.token}) + "\\n").encode())
    await writer.drain()
    await reader.readline()
    writer.write(b"not-json\\n")
    await writer.drain()
    await asyncio.sleep(1)
asyncio.run(main())
''', encoding="utf-8")
    manager = _manager(project, tmp_path)
    source = manager._source("test.host-protocol")
    identity = ExtensionContext(
        source.manifest.id, project, source_root=source.root,
        version=source.manifest.version, permissions=source.manifest.permissions,
    ).public_identity()

    async def scenario():
        host = ExtensionHost(source, identity, {}, lambda _state: None, host_script=fake_host)
        with pytest.raises(ExtensionHostError, match="Malformed extension host RPC"):
            await host.start()
        assert not host.alive

    asyncio.run(scenario())


def test_cancellation_stops_host_operation_without_blocking_main_loop(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/slow", "test.host-slow", '''import asyncio
def setup(api, ctx):
    async def slow(command_context):
        await asyncio.sleep(30)
    api.register_command("slow", title="Slow", description="Slow", handler=slow)
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-slow", require_trust=False)
        task = asyncio.create_task(commands["test.host-slow:slow"].handler({
            "operation_id": "slow-operation", "args": {}, "notebook_id": "demo",
            "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
        }))
        await asyncio.sleep(0.05)
        tick_started = time.monotonic()
        await asyncio.sleep(0.02)
        assert time.monotonic() - tick_started < 0.2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active.host.alive
        await manager.shutdown()

    asyncio.run(scenario())


def test_excessive_memory_host_is_terminated_without_blocking_unrelated_work(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/memory", "test.host-memory", '''import asyncio
def setup(api, ctx):
    async def allocate(command_context):
        value = bytearray(64 * 1024 * 1024)
        await asyncio.sleep(30)
        return len(value)
    api.register_command("allocate", title="Allocate", description="Allocate", handler=allocate)
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-memory", require_trust=False)
        active.host.memory_limit_bytes = 32 * 1024 * 1024
        active.host.monitor_interval = 0.05
        active.host.monitor_task.cancel()
        active.host.monitor_task = asyncio.create_task(active.host._monitor())
        monitor_task = active.host.monitor_task
        handler_id = active.host.contributions[0]["handler_id"]
        operation = asyncio.create_task(
            active.host.invoke("command", handler_id, {"args": {}}, timeout=5),
        )
        ticks = 0
        deadline = time.monotonic() + 2
        while not operation.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            ticks += 1
        with pytest.raises(ExtensionHostError, match="terminated|exited unexpectedly"):
            await operation
        await monitor_task
        assert ticks >= 2
        assert not active.host.alive
        await manager.shutdown()

    asyncio.run(scenario())


def test_context_permission_boundary_rejects_traversal_and_symlink_escape(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "link").symlink_to(outside, target_is_directory=True)
    denied = ExtensionContext("test.denied", project, permissions={})
    allowed = ExtensionContext("test.allowed", project, permissions={"filesystem_read": ["project"]})

    with pytest.raises(PermissionError):
        denied.resolve_project_path("file.txt")
    with pytest.raises(ValueError, match="escapes"):
        allowed.resolve_project_path("../outside")
    with pytest.raises(ValueError, match="escapes"):
        allowed.resolve_project_path("link/file.txt")


def test_core_broker_enforces_exact_permissions_and_direct_capabilities_are_denied(tmp_path):
    project = tmp_path / "project"
    allowed = project / "allowed"
    allowed.mkdir(parents=True)
    (allowed / "input.txt").write_text("brokered", encoding="utf-8")
    denied = project / "denied.txt"
    denied.write_text("secret", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "escape").symlink_to(outside, target_is_directory=True)
    _write_extension(
        project / ".dialeng/extensions/permissions",
        "test.host-permissions",
        '''import socket, subprocess
def setup(api, ctx):
    async def broker_read(command_context):
        return {"value": await ctx.read_text("allowed/input.txt")}
    async def broker_write(command_context):
        return await ctx.write_text("allowed/output.txt", "written")
    async def broker_denied(command_context):
        return {"value": await ctx.read_text("denied.txt")}
    async def broker_escape(command_context):
        return {"value": await ctx.read_text("escape/value.txt")}
    async def broker_network(command_context):
        async with ctx.open_http_client() as client:
            response = await client.get(command_context["args"]["url"])
        return {"status": response.status_code, "body": response.text}
    async def broker_process(command_context):
        return await ctx.run_process(["/bin/echo", "broker-process"])
    async def broker_notebook(command_context):
        return await ctx.change_cell_type("demo", "cell-1", "note")
    async def broker_kernel(command_context):
        return await ctx.invoke_kernel("interrupt", notebook_id="demo")
    async def broker_helpers(command_context):
        await ctx.notify("host notification", notebook_id="demo")
        return {"setting": ctx.setting("missing", "fallback")}
    def direct_read(command_context):
        return open(ctx.project_root + "/allowed/input.txt").read()
    def direct_network(command_context):
        return socket.create_connection(("127.0.0.1", command_context["args"]["port"]))
    def direct_process(command_context):
        return subprocess.run(["/bin/echo", "bypass"], capture_output=True).returncode
    for name, handler in locals().copy().items():
        if name.startswith(("broker_", "direct_")):
            api.register_command(name, title=name, description=name, handler=handler)
''',
        permissions='''filesystem_read = ["allowed/**"]
filesystem_write = ["allowed/**"]
network = true
subprocess = true
notebook = true
kernel = true
browser_assets = false''',
    )
    notifications = []
    manager = _manager(project, tmp_path, context_services={
        "change_cell_type": lambda notebook_id, cell_id, cell_type: {
            "notebook_id": notebook_id, "cell_id": cell_id, "cell_type": cell_type,
        },
        "kernel_operation": lambda name, arguments: {"name": name, **arguments},
        "notify": lambda message, **kwargs: notifications.append((message, kwargs)),
    })

    async def scenario():
        async def http_response(reader, writer):
            await reader.read(4096)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(http_response, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        active = await manager.activate("test.host-permissions", require_trust=False)
        handlers = {item["name"].rsplit(":", 1)[-1]: item["handler_id"] for item in active.host.contributions}

        async def invoke(name, args=None):
            return await active.host.invoke("command", handlers[name], {"args": args or {}})

        assert await invoke("broker_read") == {"value": "brokered"}
        assert (await invoke("broker_write"))["bytes"] == 7
        assert (allowed / "output.txt").read_text(encoding="utf-8") == "written"
        assert await invoke("broker_network", {"url": f"http://127.0.0.1:{port}/"}) == {
            "status": 200, "body": "ok",
        }
        assert (await invoke("broker_process"))["stdout"] == "broker-process\n"
        assert await invoke("broker_notebook") == {
            "notebook_id": "demo", "cell_id": "cell-1", "cell_type": "note",
        }
        assert await invoke("broker_kernel") == {"name": "interrupt", "notebook_id": "demo"}
        assert await invoke("broker_helpers") == {"setting": "fallback"}
        assert notifications == [("host notification", {"tone": "info", "notebook_id": "demo"})]
        with pytest.raises(ExtensionHostError, match="does not permit read access"):
            await invoke("broker_denied")
        with pytest.raises(ExtensionHostError, match="escapes the project root"):
            await invoke("broker_escape")
        for name, args in (
            ("direct_read", {}),
            ("direct_network", {"port": port}),
            ("direct_process", {}),
        ):
            with pytest.raises(ExtensionHostError, match="Direct .* access denied"):
                await invoke(name, args)
        server.close()
        await server.wait_closed()
        await manager.shutdown()

    asyncio.run(scenario())


def test_failed_isolated_upgrade_leaves_previous_host_and_contributions_active(tmp_path):
    project = tmp_path / "project"
    root = project / ".dialeng/extensions/upgrade"
    _write_extension(root, "test.host-upgrade", '''def setup(api, ctx):
    api.register_command("value", title="Value", description="Value", handler=lambda context: {"value": "old"})
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        previous = await manager.activate("test.host-upgrade", require_trust=False)
        (root / "extension.py").write_text(
            'def setup(api, ctx):\n    __import__("os")._exit(43)\n', encoding="utf-8",
        )
        with pytest.raises(ExtensionHostError):
            await manager.activate("test.host-upgrade", require_trust=False)
        assert manager.active["test.host-upgrade"] is previous
        assert previous.host.alive
        result = await commands["test.host-upgrade:value"].handler({
            "operation_id": "old-value", "args": {}, "notebook_id": "demo",
            "cancel_event": asyncio.Event(), "report_progress": lambda *_args, **_kwargs: None,
        })
        assert result == {"value": "old"}
        await manager.shutdown()

    asyncio.run(scenario())


def test_host_kill_during_shutdown_is_bounded_and_cleans_runtime(tmp_path):
    project = tmp_path / "project"
    _write_extension(project / ".dialeng/extensions/shutdown", "test.host-shutdown", '''def setup(api, ctx):
    def cleanup():
        __import__("os")._exit(44)
    api.add_cleanup(cleanup)
    api.register_command("noop", title="Noop", description="Noop", handler=lambda context: None)
''')
    manager = _manager(project, tmp_path)

    async def scenario():
        active = await manager.activate("test.host-shutdown", require_trust=False)
        started = time.monotonic()
        await asyncio.wait_for(manager.shutdown(), timeout=4)
        assert time.monotonic() - started < 4
        assert not active.host.alive

    asyncio.run(scenario())
