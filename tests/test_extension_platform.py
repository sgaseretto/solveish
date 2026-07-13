"""Foundation tests for the owner-aware Dialeng extension runtime."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path

import pytest
import httpx
from fasthtml.common import to_xml
from starlette.testclient import TestClient

from dialeng.core import dispatch
from dialeng.core.registry import registry
from dialeng.extensions.discovery import discover_extensions
from dialeng.extensions.manager import ExtensionManager, ExtensionNotTrustedError
from dialeng.extensions.manifest import (
    ExtensionManifestError,
    extension_source_hash,
    load_extension_manifest,
)
from dialeng.extensions.trust import ExtensionTrustStore
from dialeng.extensions.state import ExtensionStateStore
from dialeng.extensions.diagnostics import redact_text, redact_value
from dialeng.ui.extensions import ExtensionManagerModal
from dialeng.ui.command_palette import CommandPaletteModal
from dialeng.extensions import events, resources
from dialeng.services import settings_service
from dialeng.services.llm.utils import execute_tool


@contextmanager
def _registry_guard():
    maps = {
        "cell_types": dict(registry.cell_types),
        "services": dict(registry.services),
        "kernels": dict(registry.kernels),
        "providers": dict(registry.providers),
        "toolbar_items": dict(registry.toolbar_items),
        "settings_sections": dict(registry.settings_sections),
        "actions": dict(registry.actions),
        "renderers": dict(dispatch._renderers),
        "serializers": dict(dispatch._jupyter_serializers),
        "llm_converters": dict(dispatch._llm_converters),
        "settings": dict(settings_service._DYNAMIC_DEFINITIONS),
        "commands": dict(resources.commands),
        "extension_actions": dict(resources.actions),
        "extension_toolbar": dict(resources.toolbar_items),
        "shortcuts": dict(resources.shortcuts),
        "panels": dict(resources.panels),
        "status_items": dict(resources.status_items),
        "assets": dict(resources.assets),
        "tools": dict(resources.tools),
        "events": dict(events.handlers),
        "bundle_roots": dict(resources.enabled_bundle_roots),
    }
    callbacks = list(registry.callbacks)
    deserializers = list(dispatch._jupyter_deserializers)
    event_failures = dict(events.event_bus.failure_counts)
    event_disabled = set(events.event_bus.disabled_handlers)
    try:
        yield
    finally:
        registry.cell_types.clear()
        registry.cell_types.update(maps["cell_types"])
        registry.services.clear()
        registry.services.update(maps["services"])
        registry.kernels.clear()
        registry.kernels.update(maps["kernels"])
        registry.providers.clear()
        registry.providers.update(maps["providers"])
        registry.toolbar_items.clear()
        registry.toolbar_items.update(maps["toolbar_items"])
        registry.settings_sections.clear()
        registry.settings_sections.update(maps["settings_sections"])
        registry.actions.clear()
        registry.actions.update(maps["actions"])
        dispatch._renderers.clear()
        dispatch._renderers.update(maps["renderers"])
        dispatch._jupyter_serializers.clear()
        dispatch._jupyter_serializers.update(maps["serializers"])
        dispatch._llm_converters.clear()
        dispatch._llm_converters.update(maps["llm_converters"])
        settings_service._DYNAMIC_DEFINITIONS.clear()
        settings_service._DYNAMIC_DEFINITIONS.update(maps["settings"])
        resources.commands.clear()
        resources.commands.update(maps["commands"])
        resources.actions.clear()
        resources.actions.update(maps["extension_actions"])
        resources.toolbar_items.clear()
        resources.toolbar_items.update(maps["extension_toolbar"])
        resources.shortcuts.clear()
        resources.shortcuts.update(maps["shortcuts"])
        resources.panels.clear()
        resources.panels.update(maps["panels"])
        resources.status_items.clear()
        resources.status_items.update(maps["status_items"])
        resources.assets.clear()
        resources.assets.update(maps["assets"])
        resources.tools.clear()
        resources.tools.update(maps["tools"])
        resources.enabled_bundle_roots.clear()
        resources.enabled_bundle_roots.update(maps["bundle_roots"])
        events.handlers.clear()
        events.handlers.update(maps["events"])
        events.event_bus.failure_counts.clear()
        events.event_bus.failure_counts.update(event_failures)
        events.event_bus.disabled_handlers.clear()
        events.event_bus.disabled_handlers.update(event_disabled)
        registry.callbacks[:] = callbacks
        dispatch._jupyter_deserializers[:] = deserializers


def _write_extension(
    root: Path,
    source: str,
    *,
    extension_id: str = "test.owner-runtime",
    scope: str = "project",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "extension.toml").write_text(
        f'''schema_version = 1
id = "{extension_id}"
name = "Owner Runtime"
version = "0.1.0"
description = "Tests transactional extension loading."
entrypoint = "extension:setup"
dialeng_api = ">=1,<2"
scope = "{scope}"

[permissions]
network = false

[contributes]
actions = true
''',
        encoding="utf-8",
    )
    (root / "extension.py").write_text(source, encoding="utf-8")
    return root


def _manager(project: Path, extension_root: Path) -> ExtensionManager:
    return ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(project / "trust.json"),
        user_root=project / "no-user-extensions",
        development_paths=(extension_root,),
    )


def test_public_extension_import_is_acyclic():
    from dialeng.extensions import ExtensionAPI, ExtensionContext, ExtensionManager

    assert ExtensionAPI.__name__ == "ExtensionAPI"
    assert ExtensionContext.__name__ == "ExtensionContext"
    assert ExtensionManager.__name__ == "ExtensionManager"


def test_builtin_loader_does_not_include_platform_modules():
    import dialeng.core.extensions as loader

    result = {item["name"] for item in loader.list_extensions()}

    assert result == {"craft_init", "example_callbacks", "shell_cell"}
    assert not result & {"api", "manager", "manifest", "trust", "events"}


def test_builtin_extensions_use_public_owned_routes_and_inventory():
    import dialeng.app as appmod
    from dialeng.core.callbacks import Callback

    client = TestClient(appmod.app)
    response = client.post(
        "/dialeng/demo/extensions/builtin.craft_init/actions/craft_init_detect",
        json={},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"lib_name", "craft_exists"}

    inventory = appmod.extension_manager.snapshot()["extensions"]
    builtins = {item["id"]: item for item in inventory if item["origin"] == "builtin"}
    assert set(builtins) == {
        "builtin.craft_init", "builtin.example_callbacks", "builtin.shell_cell",
    }
    assert all(item["managed"] is False and item["state"] == "active" for item in builtins.values())
    callback_values = [
        contribution.value
        for owner in appmod.extension_manager.store.owners.values()
        for contribution in owner.contributions
        if contribution.kind == "registry.callback"
    ]
    assert callback_values
    assert all(isinstance(callback, Callback) for callback in callback_values)


def test_manifest_validation_and_hash_change(tmp_path):
    root = _write_extension(tmp_path / "extension", "def setup(api, ctx): pass\n")
    manifest = load_extension_manifest(root / "extension.toml")
    first_hash = extension_source_hash(root)

    assert manifest.id == "test.owner-runtime"
    assert manifest.dialeng_api == ">=1,<2"

    (root / "extension.py").write_text("def setup(api, ctx): return None\n", encoding="utf-8")
    assert extension_source_hash(root) != first_hash


def test_manifest_rejects_path_escape_and_incompatible_api(tmp_path):
    root = _write_extension(tmp_path / "extension", "def setup(api, ctx): pass\n")
    (root / "outside.js").write_text("", encoding="utf-8")
    manifest = (root / "extension.toml").read_text(encoding="utf-8")
    (root / "extension.toml").write_text(
        manifest.replace('dialeng_api = ">=1,<2"', 'dialeng_api = ">=2"')
        + '\n[assets]\njs = ["../outside.js"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ExtensionManifestError):
        load_extension_manifest(root / "extension.toml")


def test_discovery_does_not_import_extension_code(tmp_path):
    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "probe"
    marker = tmp_path / "imported.txt"
    _write_extension(
        extension,
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("imported")\n'
        'def setup(api, ctx): pass\n',
    )

    result = discover_extensions(project, user_root=tmp_path / "user")

    assert "test.owner-runtime" in result.active
    assert not marker.exists()


def test_equal_precedence_duplicate_is_diagnostic(tmp_path):
    project = tmp_path / "project"
    root = project / ".dialeng" / "extensions"
    _write_extension(root / "first", "def setup(api, ctx): pass\n")
    _write_extension(root / "second", "def setup(api, ctx): pass\n")

    result = discover_extensions(project, user_root=tmp_path / "user")

    assert "test.owner-runtime" not in result.active
    assert any("equal precedence" in item.error for item in result.diagnostics)


def test_trust_is_invalidated_by_source_or_permission_change(tmp_path):
    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "trusted"
    _write_extension(extension, "def setup(api, ctx): pass\n")
    store = ExtensionTrustStore(tmp_path / "trust.json")
    source = discover_extensions(project, user_root=tmp_path / "user").active["test.owner-runtime"]
    store.trust(source, "version")
    assert store.is_trusted(source)

    (extension / "extension.py").write_text("def setup(api, ctx): return None\n", encoding="utf-8")
    changed = discover_extensions(project, user_root=tmp_path / "user").active["test.owner-runtime"]
    assert not store.is_trusted(changed)
    assert (tmp_path / "trust.json").stat().st_mode & 0o777 == 0o600


def test_once_and_project_trust_have_distinct_source_change_semantics(tmp_path):
    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "trusted"
    _write_extension(extension, "def setup(api, ctx): pass\n")
    once = ExtensionTrustStore(tmp_path / "once.json")
    project_store = ExtensionTrustStore(tmp_path / "project.json")
    source = discover_extensions(project, user_root=tmp_path / "user").active["test.owner-runtime"]
    once.trust(source, "once")
    project_store.trust(source, "project")

    (extension / "extension.py").write_text("def setup(api, ctx): return None\n", encoding="utf-8")
    changed = discover_extensions(project, user_root=tmp_path / "user").active["test.owner-runtime"]
    assert not once.is_trusted(changed)
    assert project_store.is_trusted(changed)

    manifest = (extension / "extension.toml").read_text(encoding="utf-8")
    (extension / "extension.toml").write_text(
        manifest.replace("network = false", "network = true"), encoding="utf-8",
    )
    elevated = discover_extensions(project, user_root=tmp_path / "user").active["test.owner-runtime"]
    assert not project_store.is_trusted(elevated)


def test_corrupt_state_falls_back_and_extension_diagnostics_redact_secrets(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    store = ExtensionStateStore(path)
    assert store.enabled(tmp_path) == set()
    assert store.runtime_state(tmp_path, "test.extension") == {}

    monkeypatch.setenv("DIALENG_TEST_SECRET", "super-secret-value")
    text = redact_text("Authorization: Bearer-abc super-secret-value")
    assert "Bearer-abc" not in text
    assert "super-secret-value" not in text
    assert redact_value({"refresh_token": "raw", "nested": {"ok": "safe"}}) == {
        "refresh_token": "[REDACTED]", "nested": {"ok": "safe"},
    }


def test_untrusted_extension_cannot_activate(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(tmp_path / "extension", "def setup(api, ctx): pass\n")
    manager = _manager(project, extension)

    with pytest.raises(ExtensionNotTrustedError):
        asyncio.run(manager.activate("test.owner-runtime"))


def test_activation_is_staged_and_disable_removes_owned_contributions(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''from dialeng.core.registry import register_action

@register_action("fixture.action")
def action(nb_id, **kwargs):
    return {"version": "one"}

def setup(api, ctx):
    api.register_service("fixture.service", {"ready": True})
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        assert "fixture.action" not in registry.actions
        active = asyncio.run(manager.activate("test.owner-runtime", require_trust=False))
        assert registry.actions["fixture.action"]("demo") == {"version": "one"}
        assert registry.services["fixture.service"] == {"ready": True}
        assert {item["name"] for item in manager.snapshot()["extensions"][0]["contributions"]} == {
            "fixture.action", "fixture.service",
        }

        assert asyncio.run(manager.disable("test.owner-runtime")) is True
        assert "fixture.action" not in registry.actions
        assert "fixture.service" not in registry.services
        assert active.module_name not in __import__("sys").modules


def test_failed_reload_keeps_previous_active_version(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''from dialeng.core.registry import register_action
@register_action("fixture.reload")
def action(nb_id, **kwargs): return {"version": "one"}
def setup(api, ctx): pass
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        asyncio.run(manager.activate("test.owner-runtime", require_trust=False))
        old_module = manager.active["test.owner-runtime"].module_name
        (extension / "extension.py").write_text(
            'from dialeng.core.registry import register_action\n'
            '@register_action("fixture.reload")\n'
            'def action(nb_id, **kwargs): return {"version": "broken"}\n'
            'def setup(api, ctx): raise RuntimeError("candidate failed")\n',
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="candidate failed"):
            asyncio.run(manager.activate("test.owner-runtime", require_trust=False))

        assert registry.actions["fixture.reload"]("demo") == {"version": "one"}
        assert manager.active["test.owner-runtime"].module_name == old_module


def test_reload_removes_deleted_registration(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''from dialeng.core.registry import register_action
@register_action("fixture.removed")
def removed(nb_id, **kwargs): return {}
@register_action("fixture.kept")
def kept(nb_id, **kwargs): return {"version": "one"}
def setup(api, ctx): pass
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        asyncio.run(manager.activate("test.owner-runtime", require_trust=False))
        (extension / "extension.py").write_text(
            '''from dialeng.core.registry import register_action
@register_action("fixture.kept")
def kept(nb_id, **kwargs): return {"version": "two"}
def setup(api, ctx): pass
''',
            encoding="utf-8",
        )
        asyncio.run(manager.activate("test.owner-runtime", require_trust=False))

        assert "fixture.removed" not in registry.actions
        assert registry.actions["fixture.kept"]("demo") == {"version": "two"}


def test_repeated_reload_does_not_duplicate_callbacks_or_deserializers(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''from dialeng.core.callbacks import Callback
from dialeng.core.registry import register_callback
from dialeng.core.dispatch import register_jupyter_deserializer

@register_callback
class FixtureCallback(Callback):
    name = "FixtureCallback"

@register_jupyter_deserializer
def fixture_deserializer(cell):
    return None

def setup(api, ctx): pass
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        base_callbacks = len(registry.callbacks)
        base_deserializers = len(dispatch._jupyter_deserializers)
        for index in range(50):
            path = extension / "extension.py"
            source = path.read_text(encoding="utf-8")
            path.write_text(source + f"\n# reload {index}\n", encoding="utf-8")
            asyncio.run(manager.activate("test.owner-runtime", require_trust=False))

        assert len(registry.callbacks) == base_callbacks + 1
        assert len(dispatch._jupyter_deserializers) == base_deserializers + 1
        assert manager.active["test.owner-runtime"].reload_count == 49


def test_public_resources_are_namespaced_ordered_and_removed(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''def setup(api, ctx):
    async def hello(command_ctx=None): return {"hello": True}
    def inspect_project(path: str) -> dict:
        """Inspect one project path."""
        return {"path": path}
    api.register_command("hello", title="Hello", description="Say hello", handler=hello)
    api.register_shortcut("hello-key", "Ctrl+Alt+H", "hello")
    api.register_tool("inspect-project", inspect_project)
    api.on("notebook.saved", lambda payload: None, name="saved", order=10)
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        async def scenario():
            await manager.activate("test.owner-runtime", require_trust=False)
            assert "test.owner-runtime:hello" in resources.commands
            assert resources.shortcuts["test.owner-runtime:hello-key"].command == "test.owner-runtime:hello"
            assert "test-owner-runtime--inspect-project" in resources.tools
            assert "test.owner-runtime:notebook.saved.saved" in events.handlers

            await manager.disable("test.owner-runtime")
            assert not any(name.startswith("test.owner-runtime:") for name in resources.commands)
            assert "test-owner-runtime--inspect-project" not in resources.tools
            assert not any(name.startswith("test.owner-runtime:") for name in events.handlers)

        asyncio.run(scenario())


def test_event_bus_chains_transforms_and_guards_deterministically(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''def setup(api, ctx):
    api.on("execution.before_start", lambda payload: {**payload, "steps": payload.get("steps", []) + ["second"]}, name="second", mode="transform", order=20)
    api.on("execution.before_start", lambda payload: {**payload, "steps": payload.get("steps", []) + ["first"]}, name="first", mode="transform", order=10)
    api.on("execution.before_start", lambda payload: "blocked by fixture" if payload.get("block") else None, name="guard", mode="guard", order=30)
''',
    )
    manager = _manager(project, extension)

    with _registry_guard():
        async def scenario():
            await manager.activate("test.owner-runtime", require_trust=False)
            outcome = await events.event_bus.emit("execution.before_start", {"block": True})
            assert outcome.payload["steps"] == ["first", "second"]
            assert outcome.blocked is True
            assert outcome.reason == "blocked by fixture"
            await manager.disable("test.owner-runtime")

        asyncio.run(scenario())


def test_tool_events_transform_provider_neutral_input_and_result():
    class Registry:
        def is_builtin(self, name):
            return name == "renamed"

        async def execute_builtin(self, name, payload):
            return {"status": "success", "result": payload["value"]}

    before_name = "test.tool.before"
    after_name = "test.tool.after"
    events.handlers[before_name] = events.EventHandlerRegistration(
        before_name,
        "tool.before_call",
        lambda payload: {**payload, "tool_name": "renamed", "input": {"value": 7}},
        mode="transform",
    )
    events.handlers[after_name] = events.EventHandlerRegistration(
        after_name,
        "tool.after_call",
        lambda payload: {**payload, "result": {**payload["result"], "audited": True}},
        mode="transform",
    )
    try:
        result = asyncio.run(execute_tool("original", {}, None, "demo", Registry()))
        assert result == {"status": "success", "result": 7, "audited": True}
    finally:
        events.handlers.pop(before_name, None)
        events.handlers.pop(after_name, None)


def test_managed_tasks_and_cleanup_run_once_on_disable(tmp_path):
    project = tmp_path / "project"
    extension = _write_extension(
        tmp_path / "extension",
        '''import asyncio
def setup(api, ctx):
    async def worker():
        while True: await asyncio.sleep(10)
    ctx.state["cleanup_count"] = 0
    async def cleanup(): ctx.state["cleanup_count"] += 1
    ctx.create_task(worker(), name="worker")
    api.add_cleanup(cleanup)
''',
    )
    manager = _manager(project, extension)

    async def scenario():
            with _registry_guard():
                active = await manager.activate("test.owner-runtime", require_trust=False)
                assert active.runtime_mode == "isolated"
                assert active.host.alive
                assert await manager.disable("test.owner-runtime") is True
                assert manager.state_store.runtime_state(project, "test.owner-runtime")["cleanup_count"] == 1
                assert not active.host.alive
                assert await manager.disable("test.owner-runtime") is False

    asyncio.run(scenario())


def test_enabled_state_is_project_scoped_and_private(tmp_path):
    store = ExtensionStateStore(tmp_path / "state.json")
    first = tmp_path / "first"
    second = tmp_path / "second"

    store.set_enabled(first, "test.first", True)
    store.set_enabled(second, "test.second", True)
    store.set_enabled(first, "test.other", True)
    store.set_enabled(first, "test.first", False)

    assert store.enabled(first) == {"test.other"}
    assert store.enabled(second) == {"test.second"}
    assert (tmp_path / "state.json").stat().st_mode & 0o777 == 0o600


def test_extension_manager_routes_cover_trust_enable_action_command_and_disable(tmp_path, monkeypatch):
    import dialeng.app as appmod

    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "hello"
    _write_extension(
        extension,
        '''def setup(api, ctx):
    api.register_action("ping", lambda nb_id, **kwargs: {"pong": nb_id, **kwargs})
    api.register_command("hello", title="Hello", description="Test command", handler=lambda context: {"notebook": context["notebook_id"], "args": context["args"]})
    api.register_command("kernel-only", title="Kernel only", description="Requires a kernel", availability="kernel.required", handler=lambda context: {})
    api.register_command("bounded", title="Bounded", description="Validated", args_schema={"type": "object", "additionalProperties": False, "required": ["count"], "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 3}}}, handler=lambda context: {})
''',
        extension_id="test.route-manager",
    )
    manager = ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "user",
    )
    previous = appmod.extension_manager
    monkeypatch.setattr(appmod, "extension_manager", manager)
    client = TestClient(appmod.app)
    client.__enter__()

    try:
        inventory = client.get("/dialeng/extensions")
        assert inventory.status_code == 200
        assert inventory.json()["extensions"][0]["state"] == "blocked"

        contracts = client.get("/dialeng/extensions/contracts")
        doctor = client.get("/dialeng/extensions/doctor")
        assert contracts.status_code == 200
        assert contracts.json()["policy"]["version"] == "1.0.0"
        assert doctor.status_code == 200
        assert doctor.json()["runtime_available"] is True
        assert doctor.json()["extensions"][0]["runtime_state"] == "blocked"
        stale_surface = client.get(
            "/dialeng/demo/extensions/surfaces/status/test.route-manager/missing"
        )
        assert stale_surface.status_code == 204

        rejected = client.post("/dialeng/extensions/test.route-manager/enable")
        assert rejected.status_code == 403

        trusted = client.post(
            "/dialeng/extensions/test.route-manager/trust",
            json={"mode": "version", "enable": True},
        )
        assert trusted.status_code == 200
        assert trusted.json()["extensions"][0]["state"] == "active"
        assert manager.state_store.enabled(project) == {"test.route-manager"}

        disabled_once = client.post("/dialeng/extensions/test.route-manager/disable")
        assert disabled_once.status_code == 200
        enabled = client.post("/dialeng/extensions/test.route-manager/enable")
        assert enabled.status_code == 200
        assert enabled.json()["extensions"][0]["state"] == "active"
        assert manager.state_store.enabled(project) == {"test.route-manager"}

        catalog = client.get("/dialeng/demo/extensions/commands")
        assert catalog.status_code == 200
        by_name = {item["name"]: item for item in catalog.json()["commands"]}
        assert by_name["core:save"]["owner_label"] == "Dialeng"
        assert by_name["test.route-manager:hello"]["available"] is True
        assert by_name["test.route-manager:kernel-only"]["available"] is False
        assert "kernel" in by_name["test.route-manager:kernel-only"]["unavailable_reason"].lower()

        core_command = client.post(
            "/dialeng/demo/extensions/commands/core/open-settings",
            json={},
        )
        assert core_command.json()["client_action"] == "open-settings"

        action = client.post(
            "/dialeng/demo/extensions/test.route-manager/actions/ping",
            data={"value": "42"},
        )
        assert action.json() == {"pong": "demo", "value": "42"}

        command = client.post(
            "/dialeng/demo/extensions/commands/test.route-manager/hello",
            json={"value": 42},
        )
        assert command.json() == {"notebook": "demo", "args": {"value": 42}}

        unavailable = client.post(
            "/dialeng/demo/extensions/commands/test.route-manager/kernel-only",
            json={},
        )
        assert unavailable.status_code == 409
        assert unavailable.json()["type"] == "extension_command_unavailable"

        invalid = client.post(
            "/dialeng/demo/extensions/commands/test.route-manager/bounded",
            json={"count": True, "extra": 1},
        )
        assert invalid.status_code == 422
        assert invalid.json()["type"] == "extension_command_validation_failed"

        disabled = client.post("/dialeng/extensions/test.route-manager/disable")
        assert disabled.status_code == 200
        assert disabled.json()["extensions"][0]["state"] == "disabled"
        assert manager.state_store.enabled(project) == set()
    finally:
        client.__exit__(None, None, None)
        asyncio.run(manager.shutdown())
        monkeypatch.setattr(appmod, "extension_manager", previous)


def test_extension_manager_modal_has_accessible_backend_driven_controls():
    html = to_xml(ExtensionManagerModal())

    assert 'id="extension-manager-overlay"' in html
    assert 'role="dialog"' in html
    assert 'aria-labelledby="extension-manager-title"' in html
    assert 'id="extension-manager-list"' in html


def test_command_palette_modal_is_accessible_and_backend_driven():
    html = to_xml(CommandPaletteModal())

    assert 'id="command-palette-overlay"' in html
    assert 'role="dialog"' in html
    assert 'role="combobox"' in html
    assert 'role="listbox"' in html
    assert 'aria-modal="true"' in html


def test_long_running_extension_command_can_be_cancelled(tmp_path, monkeypatch):
    import dialeng.app as appmod

    project = tmp_path / "project"
    extension = project / ".dialeng" / "extensions" / "slow"
    _write_extension(
        extension,
        '''import asyncio
def setup(api, ctx):
    async def slow(command_context):
        await command_context["report_progress"]("Working", current=1, total=2)
        await asyncio.sleep(30)
        return {"finished": True}
    api.register_command("slow", title="Slow", description="Wait", handler=slow)
''',
        extension_id="test.slow-command",
    )
    manager = ExtensionManager(
        project,
        trust_store=ExtensionTrustStore(tmp_path / "trust.json"),
        state_store=ExtensionStateStore(tmp_path / "state.json"),
        user_root=tmp_path / "user",
    )
    monkeypatch.setattr(appmod, "extension_manager", manager)

    async def scenario():
        await manager.activate("test.slow-command", require_trust=False)
        operation_id = "ext-command-canceltest"
        transport = httpx.ASGITransport(app=appmod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post(
                "/dialeng/demo/extensions/commands/test.slow-command/slow",
                json={},
                headers={"X-Dialeng-Operation-Id": operation_id},
            ))
            await asyncio.sleep(0.05)
            cancelled = await client.post(f"/dialeng/extensions/commands/{operation_id}/cancel")
            response = await pending
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
        assert response.status_code == 200
        assert response.json()["type"] == "extension_command_cancelled"

        operation_id = "ext-command-disabletest"
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post(
                "/dialeng/demo/extensions/commands/test.slow-command/slow",
                json={},
                headers={"X-Dialeng-Operation-Id": operation_id},
            ))
            await asyncio.sleep(0.05)
            disabled = await client.post("/dialeng/extensions/test.slow-command/disable")
            response = await pending
        assert disabled.status_code == 200
        assert response.json()["type"] == "extension_command_cancelled"
        assert "test.slow-command" not in manager.active

    asyncio.run(scenario())
