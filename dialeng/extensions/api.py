"""Stable extension-facing API and managed runtime context."""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from dialeng.services.settings_service import SettingDefinition, register_dynamic_setting
from dialeng.services.tool_registry import function_to_tool_schema

from .contracts import (
    EXTENSION_API_VERSION,
    ExtensionCellType,
    ExtensionKernel,
    ExtensionProvider,
    ExtensionSetting,
)
from .context import capture_registration
from .events import EventHandlerRegistration
from .resources import (
    ActionRegistration,
    AssetRegistration,
    CommandRegistration,
    PanelRegistration,
    ShortcutRegistration,
    StatusRegistration,
    ToolRegistration,
    ToolbarRegistration,
)


Cleanup = Callable[[], Any | Awaitable[Any]]
_COMMAND_AVAILABILITY = {
    "always", "workspace", "notebook.open", "kernel.optional", "kernel.required",
}


class ExtensionContext:
    """Managed resources and stable metadata available to an extension."""

    def __init__(
        self,
        extension_id: str,
        project_root: Path,
        *,
        state=None,
        services=None,
        version: str = "",
        scope: str = "project",
        source_root: Path | None = None,
        permissions: dict[str, Any] | None = None,
    ):
        self.extension_id = extension_id
        self.project_root = Path(project_root).resolve()
        self.version = version
        self.scope = scope
        self.source_root = Path(source_root or project_root).resolve()
        self.permissions = dict(permissions or {})
        self.api_version = EXTENSION_API_VERSION
        self.logger = logging.LoggerAdapter(
            logging.getLogger(f"dialeng.extension.{extension_id}"),
            {"extension_id": extension_id},
        )
        self.state = state if state is not None else {}
        self._services = dict(services or {})
        self._cleanups: list[Cleanup] = []
        self._tasks: set[asyncio.Task] = set()
        self.cancel_signal = asyncio.Event()

    def public_identity(self) -> dict[str, Any]:
        """Return the JSON-safe identity available to isolated hosts."""

        return {
            "extension_id": self.extension_id,
            "version": self.version,
            "scope": self.scope,
            "api_version": self.api_version,
            "project_root": str(self.project_root),
            "source_root": str(self.source_root),
            "permissions": dict(self.permissions),
        }

    def add_cleanup(self, cleanup: Cleanup) -> None:
        self._cleanups.append(cleanup)

    def create_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=f"extension:{self.extension_id}:{name or 'task'}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def setting(self, path: str, default: Any = None) -> Any:
        """Read one namespaced extension setting from the active project config."""
        from dialeng.services.dialeng_config import get_config

        value: Any = get_config().raw_config
        full_path = f"extensions.{self.extension_id}.{str(path).strip('.')}"
        for part in full_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    async def change_cell_type(self, notebook_id: str, cell_id: str, cell_type: str) -> dict:
        """Change one cell type through Dialeng's backend-authoritative mutation path."""
        handler = self._services.get("change_cell_type")
        if handler is None:
            raise RuntimeError("Notebook mutation service is unavailable")
        result = handler(notebook_id, cell_id, cell_type)
        return await result if inspect.isawaitable(result) else result

    def resolve_project_path(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a path below the current project without permitting traversal."""
        candidate = (self.project_root / Path(path)).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise ValueError("Extension path escapes the project root")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def watch_path(self, path: str | Path, handler: Callable, *, name: str = "watch") -> asyncio.Task:
        """Watch one project path as a managed background task."""
        target = self.resolve_project_path(path, must_exist=True)

        async def run():
            from watchfiles import awatch
            async for changes in awatch(target):
                result = handler(changes)
                if inspect.isawaitable(result):
                    await result

        return self.create_task(run(), name=name)

    def open_http_client(self, **kwargs):
        """Create a managed HTTP client when network permission is declared."""
        if not self.permissions.get("network", False):
            raise PermissionError("Extension manifest does not declare network access")
        import httpx
        client = httpx.AsyncClient(**kwargs)
        self.add_cleanup(client.aclose)
        return client

    async def notify(
        self, message: str, *, tone: str = "info", notebook_id: str = "",
    ) -> None:
        """Show a backend-authored notification in one notebook or all open tabs."""
        handler = self._services.get("notify")
        if handler is None:
            self.logger.info("Notification: %s", message)
            return
        result = handler(str(message), tone=tone, notebook_id=notebook_id)
        if inspect.isawaitable(result):
            await result

    async def cleanup(self, timeout: float = 5.0) -> list[str]:
        errors: list[str] = []
        self.cancel_signal.set()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)
            except asyncio.TimeoutError:
                errors.append("Managed tasks did not stop before the cleanup deadline")
        for cleanup in reversed(self._cleanups):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._cleanups.clear()
        return errors


class ExtensionAPI:
    """Public registration facade used from an extension setup function."""

    def __init__(self, context: ExtensionContext):
        self.context = context

    def add_cleanup(self, cleanup: Cleanup) -> None:
        self.context.add_cleanup(cleanup)

    def _qualified(self, name: str) -> str:
        local = str(name).strip()
        if not local or ":" in local:
            raise ValueError("Extension-local names must be non-empty and cannot contain ':'")
        return f"{self.context.extension_id}:{local}"

    def register_command(
        self,
        name: str,
        *,
        title: str,
        description: str,
        handler: Callable,
        order: int = 50,
        availability: str = "always",
        args_schema: dict[str, Any] | None = None,
    ) -> str:
        availability_terms = [term.strip() for term in str(availability).split("&&")]
        invalid = [term for term in availability_terms if term not in _COMMAND_AVAILABILITY]
        if invalid:
            raise ValueError(f"Unsupported command availability predicate(s): {', '.join(invalid)}")
        qualified = self._qualified(name)
        capture_registration(
            "extension.command", qualified,
            CommandRegistration(
                qualified, title, description, handler, order, availability, args_schema or {},
            ),
        )
        return qualified

    def register_shortcut(
        self, name: str, shortcut: str, command: str, *, description: str = "",
    ) -> str:
        qualified = self._qualified(name)
        command_name = command if ":" in command else self._qualified(command)
        capture_registration(
            "extension.shortcut", qualified,
            ShortcutRegistration(qualified, shortcut, command_name, description),
        )
        return qualified

    def register_panel(
        self,
        name: str,
        *,
        title: str,
        renderer: Callable,
        location: str = "right",
        order: int = 50,
    ) -> str:
        if location not in {"left", "right", "bottom", "modal"}:
            raise ValueError(f"Unsupported panel location: {location}")
        qualified = self._qualified(name)
        capture_registration(
            "extension.panel", qualified,
            PanelRegistration(qualified, title, renderer, location, order),
        )
        return qualified

    def register_status_item(self, name: str, renderer: Callable, *, order: int = 50) -> str:
        qualified = self._qualified(name)
        capture_registration(
            "extension.status", qualified, StatusRegistration(qualified, renderer, order)
        )
        return qualified

    def register_asset(self, name: str, *, kind: str, path: str) -> str:
        if kind not in {"css", "js"}:
            raise ValueError("Extension assets must be CSS or JavaScript")
        qualified = self._qualified(name)
        capture_registration(
            "extension.asset", qualified, AssetRegistration(qualified, kind, path)
        )
        return qualified

    def register_tool(
        self, name: str, function: Callable, *, file_modifying: bool = False,
    ) -> str:
        provider_name = f"{self.context.extension_id.replace('.', '-').replace('_', '-')}--{name}"
        capture_registration(
            "extension.tool",
            provider_name,
            ToolRegistration(
                provider_name,
                function,
                function_to_tool_schema(function, provider_name),
                file_modifying,
            ),
        )
        return provider_name

    def on(
        self,
        event: str,
        handler: Callable,
        *,
        name: str | None = None,
        mode: str = "observe",
        order: int = 50,
        timeout_seconds: float = 5.0,
    ) -> str:
        if mode not in {"observe", "transform", "guard"}:
            raise ValueError(f"Unsupported event handler mode: {mode}")
        local_name = name or getattr(handler, "__name__", "handler")
        qualified = self._qualified(f"{event}.{local_name}")
        capture_registration(
            "extension.event",
            qualified,
            EventHandlerRegistration(
                qualified, event, handler, mode, order, timeout_seconds,
            ),
        )
        return qualified

    def register_callback(self, callback) -> None:
        from dialeng.core.registry import register_callback

        register_callback(callback)

    def register_renderer(self, cell_type: str, renderer: Callable) -> None:
        from dialeng.core.dispatch import register_renderer

        register_renderer(cell_type)(renderer)

    def register_llm_converter(self, cell_type: str, converter: Callable) -> None:
        from dialeng.core.dispatch import register_llm_converter

        register_llm_converter(cell_type)(converter)

    def register_service(self, name: str, service: Any) -> None:
        from dialeng.core.registry import registry

        registry.register_service(name, service)

    def register_action(
        self,
        name: str,
        handler: Callable,
        *,
        request_schema: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        qualified = self._qualified(name)
        capture_registration(
            "extension.action",
            qualified,
            ActionRegistration(qualified, handler, request_schema or {}, response_schema or {}),
        )
        return qualified

    def register_toolbar_item(
        self,
        name: str,
        renderer: Callable | None = None,
        *,
        command: str = "",
        title: str = "",
        icon: str = "blocks",
        position: str = "right",
        order: int = 50,
    ) -> str:
        if renderer is not None:
            from dialeng.core.registry import ToolbarItemRegistration, registry

            registry.register_toolbar_item(ToolbarItemRegistration(name, renderer, position, order))
            return self._qualified(name)
        qualified = self._qualified(name)
        if not command:
            raise ValueError("A dynamic toolbar item requires a command")
        command_name = command if ":" in command else self._qualified(command)
        capture_registration(
            "extension.toolbar",
            qualified,
            ToolbarRegistration(qualified, title or name.replace("-", " ").title(), command_name, icon, order),
        )
        return qualified

    def register_settings_section(
        self, name: str, label: str, renderer: Callable, *, order: int = 50,
    ) -> None:
        from dialeng.core.registry import SettingsSectionRegistration, registry

        registry.register_settings_section(SettingsSectionRegistration(name, label, renderer, order))

    def register_setting(self, definition: ExtensionSetting | SettingDefinition) -> None:
        if isinstance(definition, ExtensionSetting):
            definition = SettingDefinition(**asdict(definition))
        if not isinstance(definition, SettingDefinition):
            raise TypeError("register_setting requires ExtensionSetting")
        local_path = definition.path.strip().strip(".")
        if not local_path or local_path.startswith("extensions."):
            raise ValueError("Extension setting paths must be relative")
        if definition.apply_mode not in {"live", "new_kernel", "extension_reload", "server_restart"}:
            raise ValueError(f"Unsupported extension setting apply mode: {definition.apply_mode}")
        register_dynamic_setting(replace(
            definition,
            path=f"extensions.{self.context.extension_id}.{local_path}",
        ))

    def register_kernel(self, registration: ExtensionKernel | Any) -> None:
        from dialeng.core.registry import KernelRegistration, registry

        if isinstance(registration, ExtensionKernel):
            registration = KernelRegistration(
                registration.name,
                registration.label,
                registration.icon,
                registration.factory,
                registration.description,
                registration.requires_auth,
                list(registration.runtime_options),
            )
        if not isinstance(registration, KernelRegistration):
            raise TypeError("register_kernel requires ExtensionKernel")
        registry.register_kernel_type(registration)

    def register_provider(self, registration: ExtensionProvider | Any) -> None:
        from dialeng.core.registry import ProviderRegistration, registry

        if isinstance(registration, ExtensionProvider):
            registration = ProviderRegistration(**asdict(registration))
        if not isinstance(registration, ProviderRegistration):
            raise TypeError("register_provider requires ExtensionProvider")
        registry.register_provider(registration)

    def register_cell_type(
        self,
        registration: ExtensionCellType | Any,
        *,
        renderer: Callable | None = None,
        serializer: Callable | None = None,
        deserializer: Callable | None = None,
        llm_converter: Callable | None = None,
    ) -> None:
        from dialeng.core.registry import CellTypeRegistration, registry
        from dialeng.core.dispatch import (
            register_jupyter_deserializer,
            register_jupyter_serializer,
            register_llm_converter,
            register_renderer,
        )

        if isinstance(registration, ExtensionCellType):
            registration = CellTypeRegistration(**asdict(registration))
        if not isinstance(registration, CellTypeRegistration):
            raise TypeError("register_cell_type requires ExtensionCellType")
        registry.register_cell_type(registration)
        if renderer:
            register_renderer(registration.cell_type)(renderer)
        if serializer:
            register_jupyter_serializer(registration.cell_type)(serializer)
        if deserializer:
            register_jupyter_deserializer(deserializer)
        if llm_converter:
            register_llm_converter(registration.cell_type)(llm_converter)
