"""Backend-authoritative settings validation, snapshots, and persistence."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from dialeng.services.dialeng_config import (
    DEFAULT_CONFIG,
    DialengConfig,
    _deep_merge,
    load_config,
    save_config,
)
from dialeng.extensions.context import capture_registration


class SettingsError(ValueError):
    """Base class for settings update failures."""


class SettingsConflictError(SettingsError):
    """Raised when a stale browser attempts to update the configuration."""

    def __init__(self, current_revision: str):
        super().__init__("Settings changed in another tab or process.")
        self.current_revision = current_revision


class SettingsValidationError(SettingsError):
    """Raised with field-specific validation errors."""

    def __init__(self, errors: Mapping[str, str]):
        super().__init__("One or more settings are invalid.")
        self.errors = dict(errors)


@dataclass(frozen=True)
class SettingDefinition:
    path: str
    kind: str
    default: Any
    apply_mode: str = "live"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple[str, ...] = ()
    env_var: Optional[str] = None
    label: str = ""
    description: str = ""
    group: str = "Extension"
    order: int = 50
    secret: bool = False

    def public(self) -> dict[str, Any]:
        result = asdict(self)
        result["choices"] = list(self.choices)
        return result


_CORE_DEFINITIONS: tuple[SettingDefinition, ...] = (
    SettingDefinition("display.notebook_width", "int", 1400, "live", 600, 3000),
    SettingDefinition("display.button_size", "str", "normal", "live", choices=("compact", "normal", "large")),
    SettingDefinition("display.font_size", "int", 15, "live", 10, 24),
    SettingDefinition("display.reasoning_truncate_chars", "int", 500, "live", 0, 10000),
    SettingDefinition("aws.region", "str", "us-east-1", "live", choices=("us-east-1", "us-west-2", "eu-west-1", "eu-west-2", "ap-northeast-1")),
    SettingDefinition("models.defaults.bedrock", "str", "claude-haiku-4-5", "live"),
    SettingDefinition("models.defaults.openai_codex", "str", "gpt-5.4", "live"),
    SettingDefinition("modes.default", "str", "mock", "live", choices=("mock", "learning", "concise", "standard")),
    SettingDefinition("llm.openai_codex_enabled", "bool", True, "live"),
    SettingDefinition("llm.openai_codex_transport", "str", "auto", "live", choices=("auto", "websocket", "sse")),
    SettingDefinition("skills.enabled", "bool", True, "live"),
    SettingDefinition("skills.creator_enabled", "bool", True, "live"),
    SettingDefinition("skills.install_root", "str", "dialeng", "live", choices=("dialeng", "agents")),
    SettingDefinition("skills.link_other_root", "bool", False, "live"),
    SettingDefinition("skills.max_source_bytes", "int", 2097152, "live", 65536, 20971520),
    SettingDefinition("tool_settings.max_steps", "int", 5, "live", 1, 10),
    SettingDefinition("tool_settings.require_confirmation", "bool", False, "live"),
    SettingDefinition("tool_settings.builtin_tools_enabled", "bool", True, "live"),
    SettingDefinition("shell.shell_cells_enabled", "bool", False, "server_restart"),
    SettingDefinition("colab.enabled", "bool", False, "live"),
    SettingDefinition("colab.auth_provider", "str", "oauth_local", "server_restart", choices=("oauth_local", "oauth_remote", "adc"), env_var="DIALENG_COLAB_AUTH_PROVIDER"),
    SettingDefinition("colab.default_runtime", "str", "cpu", "new_kernel", choices=("cpu", "gpu", "tpu")),
    SettingDefinition("colab.default_accelerator", "str", "", "new_kernel"),
    SettingDefinition("thinking.max_tokens", "int", 10000, "live", 0, 50000),
    SettingDefinition("llm.debug_mode", "bool", False, "live"),
    SettingDefinition("llm.debug_log_dir", "str", "./debug_logs", "live"),
    SettingDefinition("persistence.atomic_explicit_save", "bool", True, "live"),
    SettingDefinition("persistence.autosave.enabled", "bool", False, "live"),
    SettingDefinition("persistence.autosave.debounce_ms", "int", 1500, "live", 50, 60000),
    SettingDefinition("persistence.autosave.idle_only", "bool", True, "live"),
    SettingDefinition("persistence.autosave.recovery_snapshots", "bool", True, "live"),
    SettingDefinition("persistence.autosave.max_recovery_files", "int", 5, "live", 1, 100),
)

_DYNAMIC_DEFINITIONS: dict[str, SettingDefinition] = {}
_SETTINGS_LOCK = threading.RLock()


def register_dynamic_setting(definition: SettingDefinition) -> None:
    """Register a setting rendered by an extension's settings section."""
    if capture_registration("settings.definition", definition.path, definition):
        return
    if definition.path not in {item.path for item in _CORE_DEFINITIONS}:
        _DYNAMIC_DEFINITIONS[definition.path] = definition


def get_setting_definitions(config: Optional[DialengConfig] = None) -> dict[str, SettingDefinition]:
    definitions = {item.path: item for item in _CORE_DEFINITIONS}
    definitions.update(_DYNAMIC_DEFINITIONS)
    if config is not None:
        model_providers = {
            "models.defaults.bedrock": "claudette:bedrock",
            "models.defaults.openai_codex": "openai_codex",
        }
        for path, provider in model_providers.items():
            current = definitions[path]
            definitions[path] = SettingDefinition(
                **{**asdict(current), "choices": tuple(model.id for model in config.get_models_for_provider(provider))}
            )
    return definitions


def _get_path(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def config_revision(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_raw_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SettingsValidationError({"_config": "Configuration root must be an object."})
    return value


def _coerce(definition: SettingDefinition, value: Any) -> Any:
    path = definition.path
    try:
        if definition.kind == "bool":
            if isinstance(value, bool):
                coerced = value
            elif isinstance(value, str) and value.lower() in {"true", "on", "1"}:
                coerced = True
            elif isinstance(value, str) and value.lower() in {"false", "off", "0"}:
                coerced = False
            else:
                raise ValueError("must be true or false")
        elif definition.kind == "int":
            if isinstance(value, bool):
                raise ValueError("must be an integer")
            coerced = int(value)
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("must be an integer")
            if isinstance(value, str) and str(coerced) != value.strip():
                raise ValueError("must be an integer")
        else:
            if not isinstance(value, str):
                raise ValueError("must be text")
            coerced = value.strip() if path != "llm.debug_log_dir" else value
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError({path: str(exc)}) from exc

    if definition.minimum is not None and coerced < definition.minimum:
        raise SettingsValidationError({path: f"Must be at least {definition.minimum:g}."})
    if definition.maximum is not None and coerced > definition.maximum:
        raise SettingsValidationError({path: f"Must be at most {definition.maximum:g}."})
    if definition.choices and coerced not in definition.choices:
        raise SettingsValidationError({path: f"Must be one of: {', '.join(definition.choices)}."})
    return coerced


def validate_changes(
    changes: Mapping[str, Any],
    *,
    config: Optional[DialengConfig] = None,
) -> dict[str, Any]:
    definitions = get_setting_definitions(config)
    errors: dict[str, str] = {}
    validated: dict[str, Any] = {}
    for path, value in changes.items():
        definition = definitions.get(path)
        if definition is None:
            errors[path] = "Unknown setting. Extensions must render a registered setting control first."
            continue
        if definition.env_var and os.environ.get(definition.env_var) is not None:
            errors[path] = f"Managed by environment variable {definition.env_var}."
            continue
        try:
            validated[path] = _coerce(definition, value)
        except SettingsValidationError as exc:
            errors.update(exc.errors)
    if errors:
        raise SettingsValidationError(errors)
    return validated


def settings_snapshot(
    config_path: Path,
    *,
    activation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    raw = read_raw_config(config_path)
    config = load_config(config_path=config_path, force_reload=True)
    definitions = get_setting_definitions(config)
    fields: dict[str, dict[str, Any]] = {}
    missing = object()
    for path, definition in definitions.items():
        raw_value = _get_path(raw, path, missing)
        configured = definition.default if raw_value is missing else raw_value
        source = "default" if raw_value is missing else "project"
        effective = configured
        managed = False
        diagnostic = None
        if definition.env_var and os.environ.get(definition.env_var) is not None:
            try:
                effective = _coerce(definition, os.environ[definition.env_var])
            except SettingsValidationError as exc:
                # Keep the parser's safe fallback visible while reporting the
                # invalid environment override instead of breaking the panel.
                effective = (
                    config.colab_auth_provider
                    if path == "colab.auth_provider"
                    else configured
                )
                diagnostic = next(iter(exc.errors.values()), "Invalid environment value.")
            source = "environment"
            managed = True
        public_configured = configured
        public_effective = effective
        if definition.secret:
            public_configured = "" if not configured else "********"
            public_effective = "" if not effective else "********"
        fields[path] = {
            **definition.public(),
            "configured_value": public_configured,
            "effective_value": public_effective,
            "value": public_effective if managed else public_configured,
            "is_set": bool(configured),
            "source": source,
            "managed": managed,
            "diagnostic": diagnostic,
        }
    display = {
        "notebook_width": fields["display.notebook_width"]["effective_value"],
        "button_size": fields["display.button_size"]["effective_value"],
        "font_size": fields["display.font_size"]["effective_value"],
        "reasoning_truncate_chars": fields["display.reasoning_truncate_chars"]["effective_value"],
    }
    return {
        "revision": config_revision(raw),
        "fields": fields,
        "display": display,
        "activation": dict(activation or {"status": "ready"}),
    }


def commit_settings(
    config_path: Path,
    changes: Mapping[str, Any],
    expected_revision: str,
) -> tuple[DialengConfig, dict[str, Any], list[str]]:
    """Validate and atomically commit a partial settings update."""
    with _SETTINGS_LOCK:
        current = read_raw_config(config_path)
        current_revision = config_revision(current)
        if expected_revision != current_revision:
            raise SettingsConflictError(current_revision)

        current_config = load_config(config_path=config_path, force_reload=True)
        validated = validate_changes(changes, config=current_config)
        updates: dict[str, Any] = {}
        changed_paths: list[str] = []
        for path, value in validated.items():
            if _get_path(current, path, get_setting_definitions(current_config)[path].default) == value:
                continue
            _set_path(updates, path, value)
            changed_paths.append(path)

        if not changed_paths:
            return current_config, settings_snapshot(config_path), []

        merged = _deep_merge(current, updates)
        save_config(merged, config_path)
        config = load_config(config_path=config_path, force_reload=True)
        return config, settings_snapshot(config_path), changed_paths


def apply_modes_for(paths: Iterable[str], config: Optional[DialengConfig] = None) -> list[str]:
    definitions = get_setting_definitions(config)
    order = {"live": 0, "new_kernel": 1, "extension_reload": 2, "reload": 2, "server_restart": 3}
    modes = {definitions[path].apply_mode for path in paths if path in definitions}
    return sorted(modes, key=lambda item: order.get(item, 99))
