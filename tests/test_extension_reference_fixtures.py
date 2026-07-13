import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from dialeng.core import dispatch
from dialeng.extensions.events import event_bus
from dialeng.extensions.manager import ExtensionManager
from dialeng.extensions.resources import actions, commands, status_items, toolbar_items
from dialeng.extensions.state import ExtensionStateStore
from dialeng.extensions.trust import ExtensionTrustStore
from dialeng.services.settings_service import get_setting_definitions
from dialeng.services.skill_service import SkillService


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "examples" / "extensions"


def _manager(tmp_path: Path, fixture: str) -> tuple[ExtensionManager, str]:
    project = tmp_path / "project"
    target = project / ".dialeng" / "extensions" / fixture
    target.parent.mkdir(parents=True)
    shutil.copytree(FIXTURES / fixture, target, ignore=shutil.ignore_patterns(
        "__pycache__", "*.egg-info", "dist", "build",
    ))
    manager = ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "user",
    )
    manager.refresh()
    extension_id = next(iter(manager.discovery.active))
    manager.trust_store.trust(manager.discovery.active[extension_id], "version")
    return manager, extension_id


def test_hello_workspace_public_contributions_and_persistent_state(tmp_path):
    manager, extension_id = _manager(tmp_path, "hello-workspace")

    async def scenario():
        await manager.enable(extension_id)
        command = commands[f"{extension_id}:say-hello"]
        first = await command.handler({"notebook_id": "demo", "args": {}})
        second = await command.handler({"notebook_id": "demo", "args": {}})
        assert first["message"].endswith("(1)")
        assert second["message"].endswith("(2)")
        assert f"{extension_id}:greeting" in actions
        assert f"{extension_id}:hello" in toolbar_items
        assert f"{extension_id}:invocations" in status_items
        assert f"extensions.{extension_id}.greeting" in get_setting_definitions()
        await manager.activate(extension_id)
        assert manager.active[extension_id].context.state["invocations"] == 2
        await manager.disable_persisted(extension_id)

    asyncio.run(scenario())


def test_execution_guard_affects_real_event_contract_and_cleans_task(tmp_path):
    manager, extension_id = _manager(tmp_path, "execution-guard")

    async def scenario():
        active = await manager.enable(extension_id)
        try:
            assert len(active.context._tasks) == 1
            allowed = await event_bus.emit("execution.before_start", {"source": "print('ok')"})
            blocked = await event_bus.emit("execution.before_start", {"source": "# BLOCK\nprint('no')"})
            assert not allowed.blocked
            assert blocked.blocked
            assert "blocked marker" in blocked.reason
        finally:
            await manager.disable_persisted(extension_id)
        assert not active.context._tasks

    asyncio.run(scenario())


def test_diagram_cell_round_trip_and_disabled_fallback_preserve_data(tmp_path):
    manager, extension_id = _manager(tmp_path, "diagram-cell")

    async def scenario():
        await manager.enable(extension_id)
        module = __import__(manager.active[extension_id].module_name)
        original = module.DiagramCell(id="diagram-1", source="A --> B")
        raw = dispatch.cell_to_jupyter(original)
        restored = dispatch.jupyter_to_cell(raw)
        assert str(restored.cell_type) == "diagram"
        assert restored.source == "A --> B"
        await manager.disable_persisted(extension_id)
        unsupported = dispatch.jupyter_to_cell(raw)
        saved = dispatch.cell_to_jupyter(unsupported)
        assert saved["metadata"]["dialeng_cell_type"] == "diagram"
        assert saved["source"] == "A --> B"

    asyncio.run(scenario())


def test_project_inspector_wheel_builds_with_entrypoint_and_resources(tmp_path):
    source = tmp_path / "project-inspector-package"
    shutil.copytree(
        FIXTURES / "project-inspector-package",
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "dist", "build"),
    )
    dist = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", str(source), "--out-dir", str(dist)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    wheel = next(dist.glob("*.whl"))
    script = """
import asyncio, json, tempfile
from pathlib import Path
from dialeng.extensions.discovery import discover_extensions
from dialeng.extensions.manager import ExtensionManager
from dialeng.extensions.resources import commands
project = Path(tempfile.mkdtemp())
result = discover_extensions(project, user_root=Path(tempfile.mkdtemp()))
source = result.active['dev.dialeng.project-inspector']
assert 'dialeng_project_inspector' not in __import__('sys').modules
async def run():
    manager = ExtensionManager(project, user_root=Path(tempfile.mkdtemp()))
    active = await manager.activate('dev.dialeng.project-inspector', require_trust=False)
    response = await commands['dev.dialeng.project-inspector:inspect-project'].handler({})
    await manager.shutdown()
    return {'module': active.module_name, 'command': response}
activated = asyncio.run(run())
print(json.dumps({'origin': source.origin, 'entrypoint': source.manifest.entrypoint, 'resources': source.manifest.resources, 'activated': activated}))
"""
    probe = subprocess.run(
        [
            "uv", "run", "--isolated", "--with-editable", str(ROOT),
            "--with", str(wheel), "python", "-c", script,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr
    payload = json.loads(probe.stdout.strip().splitlines()[-1])
    assert payload["origin"] == "package"
    assert payload["entrypoint"] == "dialeng_project_inspector:setup"
    assert set(payload["resources"]) == {"skills", "templates"}
    assert "files" in payload["activated"]["command"]["message"]


def test_project_inspector_resources_enable_independently_from_runtime(tmp_path):
    manager, extension_id = _manager(tmp_path, "project-inspector-package")

    manager.set_resource_enabled(extension_id, "skills", True)
    service = SkillService(manager.project_root)
    assert any(skill.name == "project-inspector" for skill in service.all_skills)
    assert extension_id not in manager.active
    assert manager.state_store.enabled_resources(manager.project_root, extension_id) == {"skills"}

    manager.set_resource_enabled(extension_id, "skills", False)
    service.refresh(force=True)
    assert not any(skill.name == "project-inspector" for skill in service.all_skills)
