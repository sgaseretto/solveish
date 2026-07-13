"""Public extension contributions that are not part of the legacy registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path


@dataclass(frozen=True)
class CommandRegistration:
    name: str
    title: str
    description: str
    handler: Callable
    order: int = 50
    availability: str = "always"
    args_schema: dict[str, Any] = field(default_factory=dict)
    category: str = "Extensions"
    keywords: tuple[str, ...] = ()
    default_shortcut: str = ""
    client_action: str = ""


@dataclass(frozen=True)
class ActionRegistration:
    name: str
    handler: Callable
    request_schema: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolbarRegistration:
    name: str
    title: str
    command: str
    icon: str = "blocks"
    order: int = 50


@dataclass(frozen=True)
class ShortcutRegistration:
    name: str
    shortcut: str
    command: str
    description: str = ""


@dataclass(frozen=True)
class PanelRegistration:
    name: str
    title: str
    renderer: Callable
    location: str = "right"
    order: int = 50


@dataclass(frozen=True)
class StatusRegistration:
    name: str
    renderer: Callable
    order: int = 50


@dataclass(frozen=True)
class AssetRegistration:
    name: str
    kind: str
    path: str


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    function: Callable
    schema: dict[str, Any]
    file_modifying: bool = False


commands: dict[str, CommandRegistration] = {}
actions: dict[str, ActionRegistration] = {}
toolbar_items: dict[str, ToolbarRegistration] = {}
shortcuts: dict[str, ShortcutRegistration] = {}
panels: dict[str, PanelRegistration] = {}
status_items: dict[str, StatusRegistration] = {}
assets: dict[str, AssetRegistration] = {}
tools: dict[str, ToolRegistration] = {}
enabled_bundle_roots: dict[tuple[str, str], tuple[Path, ...]] = {}


def bundle_roots(kind: str) -> list[tuple[str, Path]]:
    return [
        (extension_id, path)
        for (extension_id, resource_kind), paths in enabled_bundle_roots.items()
        if resource_kind == kind
        for path in paths
    ]


def public_resources() -> dict[str, list[dict[str, Any]]]:
    return {
        "commands": [
            {
                "name": value.name,
                "title": value.title,
                "description": value.description,
                "order": value.order,
                "availability": value.availability,
                "args_schema": value.args_schema,
                "category": value.category,
                "keywords": list(value.keywords),
                "default_shortcut": value.default_shortcut,
                "client_action": value.client_action,
            }
            for value in sorted(commands.values(), key=lambda item: (item.order, item.name))
        ],
        "actions": [
            {
                "name": value.name,
                "request_schema": value.request_schema,
                "response_schema": value.response_schema,
            }
            for value in actions.values()
        ],
        "toolbar_items": [value.__dict__ for value in sorted(
            toolbar_items.values(), key=lambda item: (item.order, item.name)
        )],
        "shortcuts": [value.__dict__ for value in shortcuts.values()],
        "panels": [
            {"name": value.name, "title": value.title, "location": value.location, "order": value.order}
            for value in panels.values()
        ],
        "status_items": [
            {"name": value.name, "order": value.order} for value in status_items.values()
        ],
        "assets": [value.__dict__ for value in assets.values()],
        "tools": [{"name": value.name, "file_modifying": value.file_modifying} for value in tools.values()],
    }


def _core_client_command(action: str):
    def handler(context: dict[str, Any]) -> dict[str, Any]:
        return {"client_action": action, "args": dict(context.get("args") or {})}

    return handler


def register_core_commands() -> None:
    """Register idempotent core commands in the same catalog extensions use."""

    extension_id_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["extension_id"],
        "properties": {
            "extension_id": {
                "type": "string",
                "title": "Extension ID",
                "minLength": 3,
            },
        },
    }
    definitions = (
        CommandRegistration(
            "core:open-settings", "Open Settings", "Open Dialeng project settings.",
            _core_client_command("open-settings"), 10, "workspace", {}, "View",
            ("preferences", "configuration"), "",
        ),
        CommandRegistration(
            "core:open-extensions", "Open Extensions", "Review and manage extensions.",
            _core_client_command("open-extensions"), 11, "workspace", {}, "View",
            ("plugins", "manager"), "",
        ),
        CommandRegistration(
            "core:show-shortcuts", "Show Keyboard Shortcuts", "Open the keyboard shortcut reference.",
            _core_client_command("show-shortcuts"), 12, "workspace", {}, "Help",
            ("keys", "bindings"), "?",
        ),
        CommandRegistration(
            "core:toggle-explorer", "Toggle File Explorer", "Show or hide the project file explorer.",
            _core_client_command("toggle-explorer"), 13, "workspace", {}, "View",
            ("files", "sidebar"), "Meta+Shift+E",
        ),
        CommandRegistration(
            "core:save", "Save", "Save the current notebook or text file.",
            _core_client_command("save-current"), 20, "workspace", {}, "File",
            ("write", "persist"), "Meta+S",
        ),
        CommandRegistration(
            "core:select-kernel", "Select Kernel", "Open the kernel selection dialog.",
            _core_client_command("select-kernel"), 30, "notebook.open", {}, "Kernel",
            ("attach", "local", "colab"), "",
        ),
        CommandRegistration(
            "core:add-cell", "Add Cell", "Add a cell below the focused cell.",
            _core_client_command("add-cell"), 40, "notebook.open",
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["cell_type"],
                "properties": {
                    "cell_type": {
                        "type": "string",
                        "title": "Cell type",
                        "enum": ["code", "note", "prompt"],
                        "default": "code",
                    }
                },
            },
            "Notebook", ("new", "code", "note", "prompt"), "",
        ),
        *(
            CommandRegistration(
                f"core:extension-{operation}", title,
                f"{title} for a notebook-authored extension.",
                _core_client_command(f"extension-studio:{operation}"),
                60 + index, "workspace", extension_id_schema, "Extension Development",
                ("notebook", "studio", operation), "",
            )
            for index, (operation, title) in enumerate((
                ("build", "Build Extension"),
                ("prepare", "Validate Extension"),
                ("test", "Test Extension"),
                ("preview", "Preview Extension Contributions"),
                ("package", "Package Extension"),
                ("activate-reviewed", "Activate Reviewed Extension Build"),
            ))
        ),
    )
    for registration in definitions:
        commands.setdefault(registration.name, registration)


register_core_commands()
