"""Backend-authoritative lifecycle manager for Dialeng extensions."""
from __future__ import annotations

import asyncio
import importlib.util
from importlib import metadata
import inspect
import logging
import re
import sys
import time
import uuid
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api import ExtensionAPI, ExtensionContext
from .context import RegistrationScope, registration_scope
from .contributions import ContributionStore
from .discovery import DiscoveryResult, discover_extensions
from .manifest import ExtensionSource
from .trust import ExtensionTrustStore
from .state import ExtensionStateStore


logger = logging.getLogger(__name__)


class ExtensionNotTrustedError(PermissionError):
    pass


@dataclass
class ActiveExtension:
    source: ExtensionSource
    context: ExtensionContext
    module_name: str
    loaded_at: float = field(default_factory=time.time)
    reload_count: int = 0
    cleanup_errors: list[str] = field(default_factory=list)

    def public(self, contributions: list[dict[str, str]]) -> dict[str, Any]:
        return {
            **self.source.public(),
            "state": "active",
            "loaded_at": self.loaded_at,
            "reload_count": self.reload_count,
            "cleanup_errors": list(self.cleanup_errors),
            "contributions": contributions,
            "managed_tasks": len(self.context._tasks),
        }


class ExtensionManager:
    def __init__(
        self,
        project_root: Path,
        *,
        trust_store: ExtensionTrustStore | None = None,
        user_root: Path | None = None,
        development_paths: tuple[Path, ...] = (),
        builtin_root: Path | None = None,
        state_store: ExtensionStateStore | None = None,
        contribution_store: ContributionStore | None = None,
        context_services: dict[str, Any] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.trust_store = trust_store or ExtensionTrustStore()
        self.user_root = user_root
        self.development_paths = development_paths
        self.builtin_root = builtin_root
        self.state_store = state_store or ExtensionStateStore()
        self.store = contribution_store or ContributionStore()
        self.context_services = dict(context_services or {})
        self.active: dict[str, ActiveExtension] = {}
        self.failures: dict[str, str] = {}
        self.discovery = DiscoveryResult()
        self.revision = 0
        self._lock = asyncio.Lock()

    def set_project_root(self, project_root: Path) -> None:
        if self.active:
            raise RuntimeError("Cannot change extension project root while extensions are active")
        self.project_root = Path(project_root).resolve()
        self.discovery = DiscoveryResult()

    def refresh(self) -> DiscoveryResult:
        self.discovery = discover_extensions(
            self.project_root,
            user_root=self.user_root,
            development_paths=self.development_paths,
            builtin_root=self.builtin_root,
        )
        return self.discovery

    def _source(self, extension_id: str) -> ExtensionSource:
        source = self.discovery.active.get(extension_id)
        if source is None:
            self.refresh()
            source = self.discovery.active.get(extension_id)
        if source is None:
            raise KeyError(f"Unknown extension: {extension_id}")
        return source

    @staticmethod
    def _module_name(source: ExtensionSource) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", source.manifest.id)
        return f"_dialeng_ext_{safe}_{source.source_hash[:12]}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _entrypoint_path(source: ExtensionSource) -> tuple[Path, str]:
        module_path, function_name = source.manifest.entrypoint.split(":", 1)
        relative = Path(*module_path.split("."))
        file_path = (source.root / relative).with_suffix(".py")
        package_path = source.root / relative / "__init__.py"
        if file_path.is_file():
            resolved = file_path.resolve()
        elif package_path.is_file():
            resolved = package_path.resolve()
        else:
            raise ImportError(f"Entrypoint module not found: {module_path}")
        try:
            resolved.relative_to(source.root.resolve())
        except ValueError as exc:
            raise ImportError("Entrypoint escapes extension root") from exc
        return resolved, function_name

    @staticmethod
    def _validate_candidate(source: ExtensionSource, contributions) -> None:
        contribution_keys = {
            "extension.command": "commands",
            "extension.action": "actions",
            "extension.toolbar": "toolbar",
            "extension.shortcut": "shortcuts",
            "extension.panel": "panels",
            "extension.status": "status",
            "extension.tool": "tools",
            "extension.event": "events",
            "registry.cell_type": "cell_types",
            "registry.kernel": "kernels",
            "registry.provider": "providers",
            "settings.definition": "settings",
        }
        for contribution in contributions:
            declared_key = contribution_keys.get(contribution.kind)
            if declared_key in source.manifest.contributes and not source.manifest.contributes[declared_key]:
                raise ValueError(
                    f"Manifest disables {declared_key}, but setup registered {contribution.name}"
                )
            if contribution.kind == "extension.asset":
                asset = contribution.value
                declared = source.manifest.assets.get(asset.kind, [])
                if asset.path not in declared:
                    raise ValueError(
                        f"Asset {asset.path!r} must be declared under [assets].{asset.kind}"
                    )

    async def activate(self, extension_id: str, *, require_trust: bool = True) -> ActiveExtension:
        async with self._lock:
            self.refresh()
            source = self._source(extension_id)
            if require_trust and not self.trust_store.is_trusted(source):
                raise ExtensionNotTrustedError(f"Extension {extension_id} is not trusted")
            record_path = (
                self.project_root / ".dialeng" / "extension-builds"
                / f"{hashlib.sha256(extension_id.encode()).hexdigest()}.json"
            )
            if record_path.is_file():
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Extension validation record is corrupted; run prepare again") from exc
                if record.get("source_hash") != source.source_hash:
                    raise RuntimeError("Extension changed after validation; run prepare again before activation")
            source.manifest.assert_compatible()
            function_name = source.manifest.entrypoint.split(":", 1)[1]
            module_path = None
            if source.origin != "package" or not source.package_entrypoint:
                module_path, function_name = self._entrypoint_path(source)
            module_name = self._module_name(source)
            scope = RegistrationScope(extension_id, source.manifest.version)
            context = ExtensionContext(
                extension_id,
                self.project_root,
                state=self.state_store.namespace(self.project_root, extension_id),
                services=self.context_services,
                version=source.manifest.version,
                scope=source.manifest.scope,
                source_root=source.root,
                permissions=source.manifest.permissions,
            )
            api = ExtensionAPI(context)
            module = None
            tasks_before = set(asyncio.all_tasks())
            try:
                with registration_scope(scope):
                    if source.origin == "package" and source.package_entrypoint:
                        entry_points = metadata.entry_points(group="dialeng.extensions")
                        matches = [
                            entry for entry in entry_points
                            if entry.name == source.package_entrypoint and entry.value == source.manifest.entrypoint
                        ]
                        if len(matches) != 1:
                            raise ImportError("Installed extension entry point changed since discovery")
                        setup = matches[0].load()
                        module_name = getattr(setup, "__module__", module_name)
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Could not load entrypoint module: {module_path}")
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        setup = getattr(module, function_name, None)
                    if not callable(setup):
                        raise ImportError(f"Entrypoint function not found: {function_name}")
                    result = setup(api, context)
                    if inspect.isawaitable(result):
                        await asyncio.wait_for(result, timeout=10)
                unmanaged = set()
                for task in set(asyncio.all_tasks()) - tasks_before:
                    if task is asyncio.current_task() or task in context._tasks or task.done():
                        continue
                    code = getattr(task.get_coro(), "cr_code", None)
                    filename = Path(code.co_filename).resolve() if code else None
                    if filename and filename.is_relative_to(source.root.resolve()):
                        unmanaged.add(task)
                if unmanaged:
                    for task in unmanaged:
                        task.cancel()
                    await asyncio.gather(*unmanaged, return_exceptions=True)
                    raise RuntimeError(
                        "Extension setup created unmanaged background task(s); use ctx.create_task"
                    )
                self._validate_candidate(source, scope.contributions)
                self.store.replace(extension_id, scope.contributions)
            except Exception as exc:
                await context.cleanup()
                if source.origin != "package":
                    sys.modules.pop(module_name, None)
                self.failures[extension_id] = f"{type(exc).__name__}: {exc}"
                from .events import event_bus
                await event_bus.emit("extension.failed", {
                    "extension_id": extension_id,
                    "error": self.failures[extension_id],
                })
                raise

            previous = self.active.get(extension_id)
            active = ActiveExtension(
                source=source,
                context=context,
                module_name=module_name,
                reload_count=(previous.reload_count + 1) if previous else 0,
            )
            self.active[extension_id] = active
            self.failures.pop(extension_id, None)
            self.revision += 1
            if previous:
                active.cleanup_errors = await previous.context.cleanup()
                if previous.source.origin != "package":
                    sys.modules.pop(previous.module_name, None)
            logger.info(
                "Extension activated (id=%s, version=%s, contributions=%d)",
                extension_id, source.manifest.version, len(scope.contributions),
            )
            from .events import event_bus
            await event_bus.emit("extension.reloaded" if previous else "extension.loaded", {
                "extension_id": extension_id,
                "version": source.manifest.version,
                "source_hash": source.source_hash,
                "reload_count": active.reload_count,
            })
            return active

    async def disable(self, extension_id: str) -> bool:
        async with self._lock:
            active = self.active.pop(extension_id, None)
            if active is None:
                return False
            from .events import event_bus
            await event_bus.emit("extension.unloading", {
                "extension_id": extension_id,
                "version": active.source.manifest.version,
            })
            cancel_commands = self.context_services.get("cancel_extension_commands")
            if cancel_commands is not None:
                result = cancel_commands(extension_id)
                if inspect.isawaitable(result):
                    await result
            self.store.remove(extension_id)
            active.cleanup_errors = await active.context.cleanup()
            if active.source.origin != "package":
                sys.modules.pop(active.module_name, None)
            self.revision += 1
            return True

    async def enable(self, extension_id: str) -> ActiveExtension:
        active = await self.activate(extension_id, require_trust=True)
        self.state_store.set_enabled(self.project_root, extension_id, True)
        return active

    async def disable_persisted(self, extension_id: str) -> bool:
        disabled = await self.disable(extension_id)
        self.state_store.set_enabled(self.project_root, extension_id, False)
        return disabled

    async def activate_enabled(self) -> dict[str, str]:
        """Restore trusted enabled extensions while isolating individual failures."""
        self.refresh()
        self.restore_resource_state()
        errors: dict[str, str] = {}
        for extension_id in sorted(self.state_store.enabled(self.project_root)):
            if extension_id not in self.discovery.active:
                errors[extension_id] = "Extension is enabled but no source was discovered"
                continue
            try:
                await self.activate(extension_id, require_trust=True)
            except Exception as exc:
                errors[extension_id] = f"{type(exc).__name__}: {exc}"
        self.failures.update(errors)
        return errors

    def set_resource_enabled(self, extension_id: str, kind: str, enabled: bool) -> None:
        from .resources import enabled_bundle_roots

        source = self._source(extension_id)
        if enabled and not self.trust_store.is_trusted(source):
            raise ExtensionNotTrustedError(
                f"Extension resources for {extension_id} are not trusted"
            )
        declared = source.manifest.resources.get(kind)
        if not declared:
            raise KeyError(f"Extension {extension_id} does not bundle {kind}")
        key = (extension_id, kind)
        if enabled:
            roots = tuple((source.root / path).resolve() for path in declared)
            for root in roots:
                if not root.is_relative_to(source.root.resolve()) or not root.exists():
                    raise ValueError(f"Invalid bundled {kind} resource: {root}")
            enabled_bundle_roots[key] = roots
        else:
            enabled_bundle_roots.pop(key, None)
        self.state_store.set_resource_enabled(self.project_root, extension_id, kind, enabled)
        self.revision += 1
        if kind == "skills":
            from dialeng.services.skill_service import get_skill_service
            get_skill_service().refresh(force=True)

    def restore_resource_state(self) -> None:
        for extension_id, source in self.discovery.active.items():
            enabled = self.state_store.enabled_resources(self.project_root, extension_id)
            for kind in sorted(enabled & set(source.manifest.resources)):
                try:
                    self.set_resource_enabled(extension_id, kind, True)
                except Exception as exc:
                    self.failures[f"{extension_id}:{kind}"] = f"{type(exc).__name__}: {exc}"

    async def shutdown(self) -> None:
        for extension_id in list(reversed(self.active)):
            await self.disable(extension_id)

    def snapshot(self) -> dict[str, Any]:
        from .resources import public_resources
        from .events import event_bus
        from .diagnostics import redact_value

        contribution_snapshot = self.store.snapshot()
        sources = self.discovery.active or self.refresh().active
        items = []
        for extension_id, source in sorted(sources.items()):
            active = self.active.get(extension_id)
            if active:
                item = active.public(contribution_snapshot.get(extension_id, []))
                item["trusted"] = self.trust_store.is_trusted(active.source)
                if source.source_hash != active.source.source_hash:
                    item.update({
                        "update_available": True,
                        "candidate_source_hash": source.source_hash,
                        "candidate_version": source.manifest.version,
                        "candidate_trusted": self.trust_store.is_trusted(source),
                    })
                items.append(item)
            else:
                items.append({
                    **source.public(),
                    "state": "blocked" if not self.trust_store.is_trusted(source) else "disabled",
                    "error": self.failures.get(extension_id),
                    "contributions": [],
                    "managed_tasks": 0,
                    "cleanup_errors": [],
                    "trusted": self.trust_store.is_trusted(source),
                })
            try:
                items[-1]["project_path"] = source.root.relative_to(self.project_root).as_posix()
            except ValueError:
                items[-1]["project_path"] = ""
            items[-1]["has_extension_notebook"] = (source.root / "extension.ipynb").is_file()
            module_path = source.manifest.entrypoint.split(":", 1)[0].replace(".", "/") + ".py"
            items[-1]["entrypoint_file"] = module_path
            items[-1]["bundled_resources"] = source.manifest.resources
            items[-1]["enabled_resources"] = sorted(
                self.state_store.enabled_resources(self.project_root, extension_id)
            )
        represented = {item["id"] for item in items}
        for owner, contributions in sorted(contribution_snapshot.items()):
            if owner in represented:
                continue
            is_builtin = owner.startswith("builtin.")
            is_autorun = owner.startswith("legacy.autorun.")
            if not (is_builtin or is_autorun):
                continue
            label = owner.rsplit(".", 1)[-1].replace("_", " ").replace("-", " ").title()
            items.append({
                "id": owner,
                "name": label,
                "version": "core" if is_builtin else "legacy",
                "description": (
                    "Dialeng built-in migrated to the owner-aware public extension API."
                    if is_builtin
                    else "Legacy AUTORUN extension loaded through the transactional ownership adapter. "
                         "Use `dialeng extension migrate-autorun` to create a manifest-based project extension."
                ),
                "origin": "builtin" if is_builtin else "autorun",
                "scope": "builtin" if is_builtin else "project",
                "state": "active",
                "managed": False,
                "legacy": is_autorun,
                "trusted": True,
                "contributions": contributions,
                "contributes": {},
                "assets": {},
                "permissions": {},
                "bundled_resources": {},
                "enabled_resources": [],
                "project_path": "",
                "has_extension_notebook": False,
                "entrypoint_file": "",
            })
        items.sort(key=lambda item: (item.get("origin", ""), item.get("id", "")))
        return redact_value({
            "revision": self.revision,
            "extensions": items,
            "shadowed": self.discovery.public()["shadowed"],
            "diagnostics": self.discovery.public()["diagnostics"],
            "resources": public_resources(),
            "event_diagnostics": event_bus.diagnostics(),
        })

    def doctor(self) -> dict[str, Any]:
        """Return discovery, trust, compatibility, runtime, and cleanup health."""

        from .diagnostics import build_extension_doctor

        self.refresh()
        return build_extension_doctor(
            self.project_root,
            self.discovery,
            self.trust_store,
            self.state_store,
            runtime_snapshot=self.snapshot(),
        )
