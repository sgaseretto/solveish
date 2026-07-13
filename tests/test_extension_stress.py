import asyncio
import shutil
import time
from pathlib import Path

from dialeng.document.cell import Cell, CellState
from dialeng.extensions.discovery import discover_extensions
from dialeng.extensions.events import EventHandlerRegistration, event_bus, handlers
from dialeng.extensions.manager import ExtensionManager
from dialeng.extensions.state import ExtensionStateStore
from dialeng.extensions.trust import ExtensionTrustStore
from dialeng.services.kernel.execution_queue import ExecutionQueue


ROOT = Path(__file__).parents[1]


def _write(root: Path, extension_id: str, setup_body: str = "pass") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "extension.toml").write_text(
        f'''schema_version = 1
id = "{extension_id}"
name = "Stress"
version = "0.1.0"
description = "Stress fixture."
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "project"
[permissions]
network = false
[contributes]
commands = true
''',
        encoding="utf-8",
    )
    (root / "extension.py").write_text(
        f"def setup(api, ctx):\n    {setup_body}\n", encoding="utf-8",
    )


def test_one_hundred_transactional_reloads_and_concurrent_mutations_are_leak_free(tmp_path):
    project = tmp_path / "project"
    root = project / ".dialeng" / "extensions" / "stress"
    _write(root, "test.stress", "api.register_command('only', title='Only', description='One', handler=lambda context: context)")
    manager = ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "user",
    )

    async def scenario():
        for index in range(100):
            (root / "extension.py").write_text(
                "def setup(api, ctx):\n"
                f"    api.register_command('only', title='Only {index}', description='One', handler=lambda context: context)\n",
                encoding="utf-8",
            )
            active = await manager.activate("test.stress", require_trust=False)
            assert active.reload_count == index
            assert len(manager.store.snapshot()["test.stress"]) == 1
        await asyncio.gather(
            manager.activate("test.stress", require_trust=False),
            manager.activate("test.stress", require_trust=False),
        )
        assert len(manager.store.snapshot()["test.stress"]) == 1
        await asyncio.gather(manager.disable("test.stress"), manager.disable("test.stress"))
        assert "test.stress" not in manager.store.owners

    asyncio.run(scenario())


def test_execution_guard_prevents_real_queue_kernel_call(tmp_path):
    project = tmp_path / "project"
    target = project / ".dialeng" / "extensions" / "guard"
    target.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "examples" / "extensions" / "execution-guard", target)
    manager = ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "user",
    )

    class Kernel:
        calls = 0

        async def execute_cell(self, notebook_id, cell, source=None):
            self.calls += 1
            if False:
                yield None

        def interrupt(self, notebook_id):
            return None

    async def scenario():
        await manager.activate("dev.dialeng.execution-guard", require_trust=False)
        kernel = Kernel()
        queue = ExecutionQueue(kernel)
        cell = Cell(id="blocked", source="# BLOCK\nprint('never')")
        queue.queue_cell("demo", cell)
        await asyncio.sleep(0)
        deadline = time.monotonic() + 2
        while cell.state == CellState.QUEUED and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        try:
            assert kernel.calls == 0
            assert cell.state == CellState.IDLE
            assert "Execution Guard blocked marker" in cell.output
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_repeated_failing_observer_is_isolated_and_disabled():
    name = "test.failing:event.handler"
    calls = 0

    def fail(_payload):
        nonlocal calls
        calls += 1
        raise RuntimeError("expected")

    handlers[name] = EventHandlerRegistration(name, "test.event", fail)
    try:
        for _ in range(5):
            asyncio.run(event_bus.emit("test.event", {}))
        assert calls == 3
        assert name in event_bus.disabled_handlers
        assert event_bus.failure_counts[name] == 3
    finally:
        handlers.pop(name, None)
        event_bus.reset_handler(name)


def test_disabled_manifest_discovery_remains_bounded(tmp_path):
    project = tmp_path / "project"
    for index in range(25):
        _write(
            project / ".dialeng" / "extensions" / f"fixture-{index}",
            f"test.fixture-{index}",
        )
    started = time.perf_counter()
    result = discover_extensions(project, user_root=tmp_path / "user", include_entry_points=False)
    duration = time.perf_counter() - started
    assert len(result.active) == 25
    assert duration < 2.0
