"""Versioned, transport-neutral contracts for public Dialeng extensions."""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


EXTENSION_API_VERSION = "1.0.0"
EXTENSION_API_MAJOR = 1
MANIFEST_SCHEMA_VERSION = 1
RPC_PROTOCOL_VERSION = 1
MANIFEST_CONTRIBUTION_FIELDS = (
    "actions", "callbacks", "cell_types", "commands", "events", "kernels",
    "panels", "providers", "services", "settings", "shortcuts", "skills",
    "status", "templates", "toolbar", "tools",
)
MANIFEST_PERMISSION_LIST_FIELDS = ("filesystem_read", "filesystem_write")
MANIFEST_PERMISSION_BOOL_FIELDS = (
    "network", "subprocess", "notebook", "kernel", "browser_assets",
)


@dataclass(frozen=True)
class ExtensionAPIPolicy:
    """Compatibility policy published with the extension API."""

    version: str = EXTENSION_API_VERSION
    manifest_schema: int = MANIFEST_SCHEMA_VERSION
    rpc_protocol: int = RPC_PROTOCOL_VERSION
    compatible_range: str = ">=1,<2"
    additions: str = "Minor releases may add optional fields and public methods."
    fixes: str = "Patch releases may clarify validation and fix contract defects."
    breaking: str = "Breaking changes require a new API major version."
    deprecations: str = "Deprecated APIs warn for at least one minor release before removal."

    def public(self) -> dict[str, Any]:
        return asdict(self)


API_POLICY = ExtensionAPIPolicy()


@dataclass(frozen=True)
class ExtensionSetting:
    """Transport-neutral extension setting declaration."""

    path: str
    kind: str
    default: Any
    apply_mode: str = "live"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    env_var: str | None = None
    label: str = ""
    description: str = ""
    group: str = "Extension"
    order: int = 50
    secret: bool = False


@dataclass(frozen=True)
class ExtensionKernel:
    """Privileged kernel contribution without exposing core registry classes."""

    name: str
    label: str
    icon: str
    factory: Callable
    description: str = ""
    requires_auth: bool = False
    runtime_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtensionProvider:
    """Privileged provider contribution without exposing core registry classes."""

    name: str
    label: str
    factory: Callable
    credential_checker: Callable | None = None
    priority: int = 0


@dataclass(frozen=True)
class ExtensionCellType:
    """Privileged custom-cell declaration owned by the public extension API."""

    cell_class: type
    cell_type: str
    icon: str = "file"
    label: str = ""
    description: str = ""


RPC_SAFE_API_METHODS = (
    "register_action",
    "register_asset",
    "register_command",
    "register_setting",
    "register_shortcut",
    "register_tool",
    "on",
)

PRIVILEGED_IN_PROCESS_API_METHODS = (
    "register_callback",
    "register_cell_type",
    "register_kernel",
    "register_llm_converter",
    "register_panel",
    "register_provider",
    "register_renderer",
    "register_service",
    "register_settings_section",
    "register_status_item",
    "register_toolbar_item(renderer=...)",
)

LEGACY_IMPORT_DIAGNOSTICS = {
    "dialeng.core.dispatch": "Use ExtensionAPI registration methods instead of core dispatch.",
    "dialeng.core.registry": "Use declarations exported by dialeng.extensions instead of core registry classes.",
    "fasthtml.common": "FastHTML renderers require privileged in-process execution.",
}


class ExtensionAPIDeprecationWarning(DeprecationWarning):
    """Warning category for public extension API deprecations."""


def warn_deprecated(symbol: str, replacement: str, *, removal: str) -> None:
    warnings.warn(
        f"{symbol} is deprecated; use {replacement}. It may be removed in API {removal}.",
        ExtensionAPIDeprecationWarning,
        stacklevel=2,
    )


def validate_object_schema(schema: dict[str, Any] | None, *, label: str) -> dict[str, Any]:
    """Validate the bounded JSON-Schema subset used by actions and commands."""

    value = dict(schema or {})
    if not value:
        return value
    if value.get("type", "object") != "object":
        raise ValueError(f"{label} schema root must be an object")
    properties = value.get("properties", {})
    required = value.get("required", [])
    if not isinstance(properties, dict) or any(not isinstance(item, dict) for item in properties.values()):
        raise ValueError(f"{label} schema properties must be an object of definitions")
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        raise ValueError(f"{label} schema required must be a list of property names")
    missing = sorted(set(required) - set(properties))
    if missing:
        raise ValueError(f"{label} schema requires undefined properties: {', '.join(missing)}")
    allowed_types = {"array", "boolean", "integer", "number", "object", "string"}
    invalid = sorted(
        name for name, definition in properties.items()
        if definition.get("type") not in allowed_types
    )
    if invalid:
        raise ValueError(f"{label} schema properties have unsupported types: {', '.join(invalid)}")
    return value


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def extension_contract_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema catalog for Extension API 1."""

    string_array = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    json_schema = {"type": "object"}
    permission_properties = {
        **{name: string_array for name in MANIFEST_PERMISSION_LIST_FIELDS},
        **{name: {"type": "boolean"} for name in MANIFEST_PERMISSION_BOOL_FIELDS},
    }
    definitions = {
        "manifest": _object_schema(
            {
                "schema_version": {"const": MANIFEST_SCHEMA_VERSION},
                "id": {"type": "string", "pattern": "^[a-z0-9]+(?:[._-][a-z0-9]+)+$"},
                "name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "entrypoint": {"type": "string", "pattern": "^[^:]+:[^:]+$"},
                "dialeng_api": {"type": "string", "minLength": 1},
                "scope": {"enum": ["builtin", "development", "package", "project", "user"]},
                "permissions": _object_schema(permission_properties, []),
                "contributes": _object_schema(
                    {name: {"type": "boolean"} for name in MANIFEST_CONTRIBUTION_FIELDS}, []
                ),
                "assets": _object_schema({"css": string_array, "js": string_array}, []),
                "author": {"type": "object", "additionalProperties": {"type": "string"}},
                "resources": _object_schema(
                    {
                        "skills": string_array,
                        "templates": string_array,
                        "craft": string_array,
                        "notebook_templates": string_array,
                    },
                    [],
                ),
            },
            [
                "schema_version", "id", "name", "version", "description",
                "entrypoint", "dialeng_api",
            ],
        ),
        "command": _object_schema(
            {
                "name": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "order": {"type": "integer"},
                "availability": {"type": "string"},
                "args_schema": json_schema,
                "category": {"type": "string"},
                "keywords": string_array,
                "default_shortcut": {"type": "string"},
            },
            ["name", "title", "description", "order", "availability", "args_schema"],
        ),
        "action": _object_schema(
            {
                "name": {"type": "string", "minLength": 1},
                "request_schema": json_schema,
                "response_schema": json_schema,
            },
            ["name", "request_schema", "response_schema"],
        ),
        "setting": _object_schema(
            {
                "path": {"type": "string", "minLength": 1},
                "kind": {"enum": ["bool", "int", "str"]},
                "default": {},
                "apply_mode": {
                    "enum": ["live", "new_kernel", "extension_reload", "server_restart"]
                },
                "minimum": {"type": ["number", "null"]},
                "maximum": {"type": ["number", "null"]},
                "choices": string_array,
                "env_var": {"type": ["string", "null"]},
                "label": {"type": "string"},
                "description": {"type": "string"},
                "group": {"type": "string"},
                "order": {"type": "integer"},
                "secret": {"type": "boolean"},
            },
            ["path", "kind", "default"],
        ),
        "custom_cell": _object_schema(
            {
                "cell_type": {"type": "string", "minLength": 1},
                "icon": {"type": "string"},
                "label": {"type": "string"},
                "description": {"type": "string"},
                "renderer": {"type": "string"},
                "serializer": {"type": "string"},
                "deserializer": {"type": "string"},
                "llm_converter": {"type": "string"},
            },
            ["cell_type", "icon", "label"],
        ),
        "event": _object_schema(
            {
                "name": {"type": "string", "minLength": 1},
                "event": {"type": "string", "minLength": 1},
                "mode": {"enum": ["observe", "transform", "guard"]},
                "order": {"type": "integer"},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
            },
            ["name", "event", "mode", "order", "timeout_seconds"],
        ),
        "event_payload": _object_schema(
            {
                "event": {"type": "string", "minLength": 1},
                "operation_id": {"type": "string"},
                "payload": {"type": "object"},
            },
            ["event", "payload"],
        ),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://dialeng.dev/schemas/extension-api-v1.schema.json",
        "title": "Dialeng Extension API 1 Contracts",
        "description": API_POLICY.compatible_range,
        "type": "object",
        "$defs": definitions,
    }


def write_contract_schema(path: Path) -> Path:
    """Write the deterministic schema catalog used by packages and CI."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(extension_contract_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def contract_snapshot() -> dict[str, Any]:
    return {
        "policy": API_POLICY.public(),
        "rpc_safe_methods": list(RPC_SAFE_API_METHODS),
        "privileged_in_process_methods": list(PRIVILEGED_IN_PROCESS_API_METHODS),
        "schema": extension_contract_schema(),
    }
