"""
Dialeng Configuration Service - Manages LLM configuration from dialeng_config.json.

This module handles loading, creating, and accessing the dialeng_config.json file
which controls model availability, AWS region, and other LLM-related settings.

On startup, if dialeng_config.json doesn't exist, it creates one with sensible defaults.
Users can modify this file to customize their setup.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)
SUPPORTED_MODEL_PROVIDERS = {"bedrock", "openai_codex"}
REMOVED_PROVIDER_KEYS = {
    "anthropic" + "_api",
    "claude" + "_agent_sdk",
    "claude" + "_code_subscription",
    "anthropic" + "_api_map",
    "claude" + "_code_map",
}


class ConfigValidationError(ValueError):
    """Raised when provider configuration cannot be applied safely."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("Invalid Dialeng provider configuration:\n- " + "\n- ".join(errors))

# Default configuration - used when creating new config file
DEFAULT_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "comment": "AWS region for Bedrock. Common options: us-east-1, us-west-2, eu-west-1"
    },
    "models": {
        "available": [
            {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "providers": ["bedrock"], "capabilities": ["tools", "images", "thinking", "usage"]},
            {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "providers": ["bedrock"], "capabilities": ["tools", "images", "thinking", "usage"]},
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "providers": ["bedrock"], "capabilities": ["tools", "images", "usage"]},
            {"id": "claude-3-5-haiku", "name": "Claude 3.5 Haiku", "providers": ["bedrock"], "capabilities": ["tools", "images", "usage"]},
            {"id": "gpt-5.4", "name": "GPT-5.4", "providers": ["openai_codex"], "capabilities": ["tools", "images", "thinking", "skills", "usage"]},
            {"id": "gpt-5.4-mini", "name": "GPT-5.4 Mini", "providers": ["openai_codex"], "capabilities": ["tools", "images", "thinking", "skills", "usage"]},
            {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex", "providers": ["openai_codex"], "capabilities": ["tools", "images", "thinking", "skills", "usage"]}
        ],
        "defaults": {
            "bedrock": "claude-haiku-4-5",
            "openai_codex": "gpt-5.4",
            "comment": "Default model for each supported provider."
        },
        "bedrock_map": {
            "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
            "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "comment": "Model IDs for AWS Bedrock with cross-region inference. Format: us.anthropic.{model}-{date}-v{n}:{profile}"
        },
        "openai_codex_map": {
            "gpt-5.4": "gpt-5.4",
            "gpt-5.4-mini": "gpt-5.4-mini",
            "gpt-5.3-codex": "gpt-5.3-codex",
            "comment": "Models exposed by the OpenAI Codex subscription transport"
        }
    },
    "modes": {
        "default": "mock",
        "comment": "Default dialog mode when opening a notebook. Options: mock, learning, concise, standard"
    },
    "thinking": {
        "max_tokens": 10000,
        "comment": "Maximum tokens for provider-supported reasoning. Set to 0 to disable."
    },
    "llm": {
        "debug_mode": False,
        "debug_log_dir": "./debug_logs",
        "openai_codex_enabled": True,
        "openai_codex_transport": "auto",
        "comment": "LLM provider settings. debug_mode saves prompts and responses to JSON files."
    },
    "skills": {
        "enabled": True,
        "creator_enabled": True,
        "install_root": "dialeng",
        "link_other_root": False,
        "max_source_bytes": 2097152,
        "comment": "Load project skills and allow reviewable notebook-to-skill creation."
    },
    "persistence": {
        "atomic_explicit_save": True,
        "autosave": {
            "enabled": False,
            "debounce_ms": 1500,
            "idle_only": True,
            "recovery_snapshots": True,
            "max_recovery_files": 5
        },
        "comment": "Explicit saves are atomic. Autosave and recovery files are opt-in and disabled by default."
    },
    "tool_settings": {
        "max_steps": 5,
        "require_confirmation": False,
        "builtin_tools_enabled": True,
        "comment": "Tool calling settings. max_steps: max tool calls per prompt (1-10). require_confirmation: prompt before file-modifying tools. builtin_tools_enabled: always-available file tools (view, rg, create, str_replace, insert)."
    },
    "display": {
        "reasoning_truncate_chars": 500,
        "notebook_width": 1400,
        "button_size": "normal",
        "font_size": 15,
        "comment": "Display settings. reasoning_truncate_chars: max chars for reasoning (0=no limit). notebook_width: max container width in px. button_size: normal/compact/large. font_size: base font size in px."
    },
    "shell": {
        "shell_cells_enabled": False,
        "comment": "Shell settings. shell_cells_enabled: if true, show dedicated Shell cell type in UI. If false, use %bash/%% bash magic or !command in code cells instead."
    },
    "colab": {
        "enabled": False,
        "auth_provider": "oauth_local",
        "default_runtime": "cpu",
        "default_accelerator": "",
        "comment": "Google Colab integration. auth_provider options: oauth_local, oauth_remote, adc. DIALENG_COLAB_AUTH_PROVIDER overrides this value."
    }
}


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    providers: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class DialengConfig:
    """Parsed dialeng configuration."""
    # AWS settings
    aws_region: str = "us-east-1"

    # Available models for UI picker
    available_models: List[ModelConfig] = field(default_factory=list)

    # Model ID mappings for different backends
    bedrock_map: Dict[str, str] = field(default_factory=dict)
    openai_codex_map: Dict[str, str] = field(default_factory=dict)

    # Provider-specific default models
    default_models: Dict[str, str] = field(default_factory=dict)

    # Default mode
    default_mode: str = "mock"

    # Extended thinking settings
    thinking_max_tokens: int = 10000

    # LLM provider settings
    debug_mode: bool = False  # Enable debug logging to files
    debug_log_dir: str = "./debug_logs"  # Directory for debug logs
    openai_codex_enabled: bool = True
    openai_codex_transport: str = "auto"

    # Project-scoped skill settings
    skills_enabled: bool = True
    skills_creator_enabled: bool = True
    skills_install_root: str = "dialeng"
    skills_link_other_root: bool = False
    skills_max_source_bytes: int = 2 * 1024 * 1024

    # Persistence. Autosave remains opt-in.
    persistence_atomic_explicit_save: bool = True
    autosave_enabled: bool = False
    autosave_debounce_ms: int = 1500
    autosave_idle_only: bool = True
    autosave_recovery_snapshots: bool = True
    autosave_max_recovery_files: int = 5

    # Tool calling settings
    tool_max_steps: int = 5  # Maximum tool calls per prompt
    tool_require_confirmation: bool = False  # Require confirmation for file-modifying tools
    tool_builtin_enabled: bool = True  # Enable built-in file tools (view, rg, etc.)

    # Display settings
    reasoning_truncate_chars: int = 500  # Max chars for reasoning text before truncation (0 = no limit)
    display_notebook_width: int = 1400  # Max container width in px
    display_button_size: str = "normal"  # Button size: normal, compact, large
    display_font_size: int = 15  # Base font size in px

    # Shell settings
    shell_cells_enabled: bool = False  # If true, show dedicated Shell cell type in UI

    # Colab settings
    colab_enabled: bool = False
    colab_auth_provider: str = "oauth_local"
    colab_default_runtime: str = "cpu"
    colab_default_accelerator: str = ""

    # Raw config for reference
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def get_default_model(self, backend: Optional[str] = None) -> str:
        """Get the default model ID for a given backend.

        Args:
            backend: The provider backend - "bedrock", "openai_codex", or None.

        Returns:
            The default model ID for the specified backend.
        """
        backend = self.provider_backend(backend)
        if backend and backend in self.default_models:
            return self.default_models[backend]

        # Use fallback if backend not found or not specified
        fallback = self.default_models.get("fallback", "gpt-5.4")
        return fallback

    @staticmethod
    def provider_backend(provider: Optional[str]) -> Optional[str]:
        """Normalize a provider selection key to its model-config backend."""
        aliases = {
            "e2e_fake": "e2e_fake",
            "claudette:bedrock": "bedrock",
            "openai_codex": "openai_codex",
        }
        return aliases.get(provider or "", provider)

    def get_models_for_provider(self, provider: Optional[str]) -> List[ModelConfig]:
        """Return models compatible with a provider selection key."""
        backend = self.provider_backend(provider)
        if not backend:
            return list(self.available_models)
        return [m for m in self.available_models if not m.providers or backend in m.providers]

    def get_model_choices(self, provider: Optional[str] = None) -> List[tuple]:
        """Get model choices for UI select (id, name) tuples."""
        return [(m.id, m.name) for m in self.get_models_for_provider(provider)]

    def validate_model(self, provider: Optional[str], model_id: Optional[str]) -> str:
        """Return a compatible model ID, falling back to the provider default."""
        valid = {m.id for m in self.get_models_for_provider(provider)}
        if model_id and model_id in valid:
            return model_id
        default = self.get_default_model(provider)
        if default in valid:
            return default
        return next(iter(valid), default)

    def get_api_model_name(self, model_id: str, backend: str) -> str:
        """Get the API model name for a given model ID and backend.

        Args:
            model_id: The UI model ID (e.g., "claude-sonnet-4-5")
            backend: "bedrock" or "openai_codex"

        Returns:
            The appropriate model name for the API
        """
        backend = self.provider_backend(backend)
        if backend == "e2e_fake":
            return model_id
        if backend == "bedrock":
            return self.bedrock_map.get(model_id, model_id)
        elif backend == "openai_codex":
            return self.openai_codex_map.get(model_id, model_id)
        raise ValueError(f"Unsupported LLM backend: {backend}")


# Module-level cached config
_config: Optional[DialengConfig] = None
_config_path: Optional[Path] = None


def _parse_config(raw: Dict[str, Any]) -> DialengConfig:
    """Parse raw JSON config into DialengConfig."""
    _validate_provider_config(raw)
    config = DialengConfig(raw_config=raw)

    # AWS settings
    aws = raw.get("aws", {})
    config.aws_region = aws.get("region", "us-east-1")

    # Models
    models = raw.get("models", {})

    # Available models
    available = models.get("available", [])
    config.available_models = [
        ModelConfig(
            id=m.get("id", ""),
            name=m.get("name", m.get("id", "")),
            providers=list(m.get("providers") or (
                ["openai_codex"] if str(m.get("id", "")).startswith("gpt-")
                else ["bedrock"]
            )),
            capabilities=list(m.get("capabilities", [])),
        )
        for m in available
        if m.get("id")  # Skip entries without ID
    ]

    # Provider-specific default models
    defaults = models.get("defaults", {})
    config.default_models = {
        k: v for k, v in defaults.items()
        if k != "comment"
    }

    # Model mappings (skip "comment" keys)
    config.bedrock_map = {
        k: v for k, v in models.get("bedrock_map", {}).items()
        if k != "comment"
    }
    config.openai_codex_map = {
        k: v for k, v in models.get("openai_codex_map", {}).items()
        if k != "comment"
    }

    # Modes
    modes = raw.get("modes", {})
    config.default_mode = modes.get("default", "mock")

    # Thinking settings
    thinking = raw.get("thinking", {})
    config.thinking_max_tokens = thinking.get("max_tokens", 10000)

    # LLM provider settings
    llm = raw.get("llm", {})
    config.debug_mode = llm.get("debug_mode", False)
    config.debug_log_dir = llm.get("debug_log_dir", "./debug_logs")
    config.openai_codex_enabled = bool(llm.get("openai_codex_enabled", True))
    config.openai_codex_transport = str(llm.get("openai_codex_transport", "auto"))
    if config.openai_codex_transport not in {"auto", "websocket", "sse"}:
        logger.warning("Invalid Codex transport %r; using auto", config.openai_codex_transport)
        config.openai_codex_transport = "auto"

    skills = raw.get("skills", {})
    config.skills_enabled = bool(skills.get("enabled", True))
    config.skills_creator_enabled = bool(skills.get("creator_enabled", True))
    config.skills_install_root = skills.get("install_root", "dialeng")
    if config.skills_install_root not in ("dialeng", "agents"):
        raise ValueError("skills.install_root must be 'dialeng' or 'agents'")
    config.skills_link_other_root = skills.get("link_other_root", False)
    if not isinstance(config.skills_link_other_root, bool):
        raise ValueError("skills.link_other_root must be a boolean")
    config.skills_max_source_bytes = max(
        64 * 1024,
        min(int(skills.get("max_source_bytes", 2 * 1024 * 1024)), 20 * 1024 * 1024),
    )

    persistence = raw.get("persistence", {})
    autosave = persistence.get("autosave", {})
    config.persistence_atomic_explicit_save = bool(
        persistence.get("atomic_explicit_save", True)
    )
    config.autosave_enabled = bool(autosave.get("enabled", False))
    config.autosave_debounce_ms = max(50, int(autosave.get("debounce_ms", 1500)))
    config.autosave_idle_only = bool(autosave.get("idle_only", True))
    config.autosave_recovery_snapshots = bool(
        autosave.get("recovery_snapshots", True)
    )
    config.autosave_max_recovery_files = max(
        1,
        int(autosave.get("max_recovery_files", 5)),
    )

    # Tool calling settings
    tool_settings = raw.get("tool_settings", {})
    config.tool_max_steps = tool_settings.get("max_steps", 5)
    config.tool_require_confirmation = tool_settings.get("require_confirmation", False)
    config.tool_builtin_enabled = tool_settings.get("builtin_tools_enabled", True)

    # Display settings
    display = raw.get("display", {})
    config.reasoning_truncate_chars = display.get("reasoning_truncate_chars", 500)
    config.display_notebook_width = display.get("notebook_width", 1400)
    config.display_button_size = display.get("button_size", "normal")
    config.display_font_size = display.get("font_size", 15)

    # Shell settings
    shell = raw.get("shell", {})
    config.shell_cells_enabled = shell.get("shell_cells_enabled", False)

    # Colab settings
    colab = raw.get("colab", {})
    config.colab_enabled = colab.get("enabled", False)
    config.colab_auth_provider = os.environ.get(
        "DIALENG_COLAB_AUTH_PROVIDER",
        colab.get("auth_provider", "oauth_local"),
    )
    if config.colab_auth_provider not in {"oauth_local", "oauth_remote", "adc"}:
        logger.warning(
            "Invalid Colab auth provider %r; falling back to oauth_local",
            config.colab_auth_provider,
        )
        config.colab_auth_provider = "oauth_local"
    config.colab_default_runtime = colab.get("default_runtime", "cpu")
    if config.colab_default_runtime not in {"cpu", "gpu", "tpu"}:
        logger.warning(
            "Invalid Colab default runtime %r; falling back to cpu",
            config.colab_default_runtime,
        )
        config.colab_default_runtime = "cpu"
    config.colab_default_accelerator = str(colab.get("default_accelerator", "") or "")

    return config


def _validate_provider_config(raw: Dict[str, Any]) -> None:
    """Reject removed or unknown provider keys instead of silently ignoring them."""
    errors: List[str] = []
    models = raw.get("models") or {}
    llm = raw.get("llm") or {}

    for key in sorted(REMOVED_PROVIDER_KEYS & (set(raw) | set(models) | set(llm))):
        errors.append(
            f"{key!r} was removed. Supported LLM providers are openai_codex and bedrock."
        )

    supported = set(SUPPORTED_MODEL_PROVIDERS)
    if os.environ.get("DIALENG_E2E") == "1":
        supported.add("e2e_fake")
    defaults = models.get("defaults") or {}
    unknown_defaults = set(defaults) - supported - {"comment"}
    for key in sorted(unknown_defaults):
        errors.append(
            f"models.defaults.{key} is unsupported; use 'bedrock' or 'openai_codex'."
        )

    for index, model in enumerate(models.get("available") or []):
        providers = set(model.get("providers") or [])
        if not providers:
            errors.append(
                f"models.available[{index}].providers is required; choose bedrock or openai_codex."
            )
        unknown = providers - supported
        if unknown:
            errors.append(
                f"models.available[{index}].providers contains unsupported values: "
                f"{', '.join(sorted(unknown))}."
            )

    if errors:
        raise ConfigValidationError(errors)


def _create_default_config(config_path: Path) -> Dict[str, Any]:
    """Create default config file and return the config dict."""
    logger.info(f"Creating default dialeng_config.json at {config_path}")

    # Write with nice formatting
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    print("   Created dialeng_config.json with defaults")
    return DEFAULT_CONFIG


def load_config(config_path: Optional[Path] = None, force_reload: bool = False,
                create_if_missing: bool = True) -> DialengConfig:
    """
    Load dialeng configuration from JSON file.

    Args:
        config_path: Path to config file. Uses cached path or env var.
        force_reload: If True, reload from disk even if cached
        create_if_missing: If True, create default config file when it doesn't
            exist. Set to False for early/module-level loads where the final
            path isn't known yet (avoids creating config in the wrong directory).

    Returns:
        Parsed DialengConfig
    """
    global _config, _config_path

    # Import-time callers do not yet know the CLI project root. Reading a
    # coincidental config from the process CWD can validate or cache the wrong
    # project, so bootstrap from defaults until an explicit path is supplied.
    if (
        config_path is None
        and not create_if_missing
        and not os.environ.get("DIALENG_CONFIG_PATH")
        and _config_path is None
    ):
        return _parse_config(DEFAULT_CONFIG)

    if config_path is None:
        # Priority: env var > cached path from previous load > CWD fallback
        env_path = os.environ.get("DIALENG_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        elif _config_path is not None:
            config_path = _config_path
        else:
            config_path = Path.cwd() / "dialeng_config.json"

    # Return cached if available and path matches
    if _config is not None and not force_reload and _config_path == config_path:
        return _config

    _config_path = config_path

    if not config_path.exists():
        if create_if_missing:
            raw = _create_default_config(config_path)
        else:
            # Use defaults in memory without writing to disk
            raw = DEFAULT_CONFIG
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            logger.info(f"Loaded dialeng_config.json from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dialeng_config.json: {e}")
            print("   Warning: Invalid dialeng_config.json, using defaults")
            raw = DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Failed to load dialeng_config.json: {e}")
            raw = DEFAULT_CONFIG

    _config = _parse_config(raw)
    return _config


def get_config() -> DialengConfig:
    """Get the current config, loading if necessary."""
    if _config is None:
        return load_config()
    return _config


def reset_config_cache() -> None:
    """Reset cached config (useful for testing)."""
    global _config, _config_path
    _config = None
    _config_path = None


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge updates into base dict.

    Args:
        base: Base dictionary to merge into
        updates: Updates to apply (nested dicts are merged, not replaced)

    Returns:
        New merged dictionary
    """
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config_dict: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """Save a complete config dictionary to the config file.

    Args:
        config_dict: Complete configuration dictionary to save
        config_path: Path to config file. Uses cached path from load_config()

    Raises:
        IOError: If the file cannot be written
    """
    global _config, _config_path

    if config_path is None:
        if _config_path is None:
            raise RuntimeError("Config path not set — call load_config() with explicit path first")
        config_path = _config_path

    logger.info(f"Saving config to {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(config_dict, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, config_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    # Reload the config to update cache
    _config = _parse_config(config_dict)
    _config_path = config_path


def update_config(updates: Dict[str, Any], config_path: Optional[Path] = None) -> DialengConfig:
    """Update specific keys in the config file.

    Performs a deep merge, so nested dictionaries are merged rather than replaced.
    For example, updating {"tool_settings": {"max_steps": 10}} will only change
    max_steps while preserving other tool_settings values.

    Args:
        updates: Dictionary of updates to apply (can be nested)
        config_path: Path to config file. Uses cached path from load_config()

    Returns:
        Updated DialengConfig after applying changes

    Raises:
        IOError: If the file cannot be read or written

    Example:
        >>> update_config({"tool_settings": {"max_steps": 10}})
        >>> update_config({"modes": {"default": "learning"}})
        >>> update_config({
        ...     "models": {"defaults": {"bedrock": "claude-sonnet-4-5"}},
        ...     "display": {"reasoning_truncate_chars": 1000}
        ... })
    """
    global _config, _config_path

    if config_path is None:
        if _config_path is None:
            raise RuntimeError("Config path not set — call load_config() with explicit path first")
        config_path = _config_path

    # Load current config from file (not cache, to get raw dict)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            current = json.load(f)
    else:
        current = DEFAULT_CONFIG.copy()

    # Deep merge updates
    merged = _deep_merge(current, updates)

    # Save and reload
    save_config(merged, config_path)

    logger.info(f"Updated config with: {list(updates.keys())}")
    return _config


def get_config_dict() -> Dict[str, Any]:
    """Get the raw config dictionary for the settings UI.

    Returns:
        Raw config dictionary as loaded from file
    """
    config = get_config()
    return config.raw_config


def print_config_status(config: DialengConfig, detected_backend: Optional[str] = None) -> None:
    """Print config status for startup logging.

    Args:
        config: The parsed DialengConfig
        detected_backend: The detected backend from credential detection (for showing active default)
    """
    models = ", ".join(m.name for m in config.available_models)
    print("   Config: dialeng_config.json")
    print(f"      AWS Region:     {config.aws_region}")
    print(f"      Models:         {models}")

    # Show default models per provider
    bedrock_default = config.get_default_model("bedrock")
    codex_default = config.get_default_model("openai_codex")

    # Highlight the active default based on detected backend
    if detected_backend == "bedrock":
        print(f"      Default Model:  {bedrock_default} (Bedrock) ← active")
        print(f"                      {codex_default} (OpenAI Codex)")
    elif detected_backend == "openai_codex":
        print(f"      Default Model:  {bedrock_default} (Bedrock)")
        print(f"                      {codex_default} (OpenAI Codex) ← active")
    else:
        print(f"      Default Model:  {bedrock_default} (Bedrock)")
        print(f"                      {codex_default} (OpenAI Codex)")

    print(f"      Default Mode:   {config.default_mode}")
    if config.debug_mode:
        print(f"      Debug Mode:     ON (logs to {config.debug_log_dir})")
    # Tool settings
    tools_status = "enabled" if config.tool_builtin_enabled else "disabled"
    confirm_status = "required" if config.tool_require_confirmation else "off"
    print(f"      Tool Calling:   {tools_status} (max {config.tool_max_steps} steps, confirm: {confirm_status})")
