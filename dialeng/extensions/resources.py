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
