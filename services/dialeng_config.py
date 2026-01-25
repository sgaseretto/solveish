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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Default configuration - used when creating new config file
DEFAULT_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "comment": "AWS region for Bedrock. Common options: us-east-1, us-west-2, eu-west-1"
    },
    "models": {
        "available": [
            {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"},
            {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku", "name": "Claude 3.5 Haiku"}
        ],
        "defaults": {
            "bedrock": "claude-haiku-4-5",
            "anthropic_api": "claude-sonnet-4-5",
            "claude_code_subscription": "claude-sonnet-4-5",
            "fallback": "claude-sonnet-4-5",
            "comment": "Default model per provider. bedrock=AWS Bedrock, anthropic_api=direct API, claude_code_subscription=Claude Code CLI. fallback is used when provider is unknown."
        },
        "anthropic_api_map": {
            "claude-haiku-4-5": "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250514",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
            "comment": "Model IDs for direct Anthropic API (with date suffix)"
        },
        "bedrock_map": {
            "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
            "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-5-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "comment": "Model IDs for AWS Bedrock with cross-region inference. Format: us.anthropic.{model}-{date}-v{n}:{profile}"
        },
        "claudette_agent_map": {
            "claude-haiku-4-5": "haiku",
            "claude-sonnet-4-5": "sonnet",
            "claude-3-5-sonnet": "sonnet",
            "claude-3-5-haiku": "haiku",
            "comment": "Model IDs for Claude Code subscription - uses simple names (sonnet, haiku, opus)"
        }
    },
    "modes": {
        "default": "mock",
        "comment": "Default dialog mode when opening a notebook. Options: mock, learning, concise, standard"
    },
    "thinking": {
        "max_tokens": 10000,
        "comment": "Maximum tokens for extended thinking. Set to 0 to disable. Requires thinking-capable model (Claude Sonnet 3.7+, Sonnet 4+, Opus 4+)"
    },
    "llm": {
        "use_sdk_directly": False,
        "debug_mode": False,
        "debug_log_dir": "./debug_logs",
        "comment": "LLM provider settings. use_sdk_directly=true uses claude-agent-sdk directly for maximum isolation (stateless). Set to false (default) to use claudette-agent wrapper."
    },
    "tool_settings": {
        "max_steps": 5,
        "require_confirmation": False,
        "builtin_tools_enabled": True,
        "comment": "Tool calling settings. max_steps: max tool calls per prompt (1-10). require_confirmation: prompt before file-modifying tools. builtin_tools_enabled: always-available file tools (view, rg, create, str_replace, insert)."
    },
    "display": {
        "reasoning_truncate_chars": 500,
        "comment": "Display settings. reasoning_truncate_chars: max characters for LLM reasoning text before truncation (0 = no limit)."
    }
}


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str


@dataclass
class DialengConfig:
    """Parsed dialeng configuration."""
    # AWS settings
    aws_region: str = "us-east-1"

    # Available models for UI picker
    available_models: List[ModelConfig] = field(default_factory=list)

    # Model ID mappings for different backends
    anthropic_api_map: Dict[str, str] = field(default_factory=dict)
    bedrock_map: Dict[str, str] = field(default_factory=dict)
    claudette_agent_map: Dict[str, str] = field(default_factory=dict)

    # Provider-specific default models
    default_models: Dict[str, str] = field(default_factory=dict)

    # Default mode
    default_mode: str = "mock"

    # Extended thinking settings
    thinking_max_tokens: int = 10000

    # LLM provider settings
    use_sdk_directly: bool = False  # Use claude-agent-sdk directly for maximum isolation
    debug_mode: bool = False  # Enable debug logging to files
    debug_log_dir: str = "./debug_logs"  # Directory for debug logs

    # Tool calling settings
    tool_max_steps: int = 5  # Maximum tool calls per prompt
    tool_require_confirmation: bool = False  # Require confirmation for file-modifying tools
    tool_builtin_enabled: bool = True  # Enable built-in file tools (view, rg, etc.)

    # Display settings
    reasoning_truncate_chars: int = 500  # Max chars for reasoning text before truncation (0 = no limit)

    # Raw config for reference
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def get_default_model(self, backend: Optional[str] = None) -> str:
        """Get the default model ID for a given backend.

        Args:
            backend: The provider backend - "bedrock", "anthropic_api", "claude_code_subscription",
                    or None to use fallback.

        Returns:
            The default model ID for the specified backend.
        """
        if backend and backend in self.default_models:
            return self.default_models[backend]

        # Use fallback if backend not found or not specified
        fallback = self.default_models.get("fallback", "claude-sonnet-4-5")
        return fallback

    def get_model_choices(self) -> List[tuple]:
        """Get model choices for UI select (id, name) tuples."""
        return [(m.id, m.name) for m in self.available_models]

    def get_api_model_name(self, model_id: str, backend: str) -> str:
        """Get the API model name for a given model ID and backend.

        Args:
            model_id: The UI model ID (e.g., "claude-sonnet-4-5")
            backend: "anthropic_api", "bedrock", or "claude_code_subscription"

        Returns:
            The appropriate model name for the API
        """
        if backend == "bedrock":
            return self.bedrock_map.get(model_id, model_id)
        elif backend == "claude_code_subscription":
            return self.claudette_agent_map.get(model_id, model_id)
        else:
            return self.anthropic_api_map.get(model_id, model_id)


# Module-level cached config
_config: Optional[DialengConfig] = None
_config_path: Optional[Path] = None


def _parse_config(raw: Dict[str, Any]) -> DialengConfig:
    """Parse raw JSON config into DialengConfig."""
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
            name=m.get("name", m.get("id", ""))
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
    config.anthropic_api_map = {
        k: v for k, v in models.get("anthropic_api_map", {}).items()
        if k != "comment"
    }
    config.bedrock_map = {
        k: v for k, v in models.get("bedrock_map", {}).items()
        if k != "comment"
    }
    config.claudette_agent_map = {
        k: v for k, v in models.get("claudette_agent_map", {}).items()
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
    config.use_sdk_directly = llm.get("use_sdk_directly", False)
    config.debug_mode = llm.get("debug_mode", False)
    config.debug_log_dir = llm.get("debug_log_dir", "./debug_logs")

    # Tool calling settings
    tool_settings = raw.get("tool_settings", {})
    config.tool_max_steps = tool_settings.get("max_steps", 5)
    config.tool_require_confirmation = tool_settings.get("require_confirmation", False)
    config.tool_builtin_enabled = tool_settings.get("builtin_tools_enabled", True)

    # Display settings
    display = raw.get("display", {})
    config.reasoning_truncate_chars = display.get("reasoning_truncate_chars", 500)

    return config


def _create_default_config(config_path: Path) -> Dict[str, Any]:
    """Create default config file and return the config dict."""
    logger.info(f"Creating default dialeng_config.json at {config_path}")

    # Write with nice formatting
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    print(f"   Created dialeng_config.json with defaults")
    return DEFAULT_CONFIG


def load_config(config_path: Optional[Path] = None, force_reload: bool = False) -> DialengConfig:
    """
    Load dialeng configuration from JSON file.

    Creates default config if file doesn't exist.

    Args:
        config_path: Path to config file. Defaults to ./dialeng_config.json
        force_reload: If True, reload from disk even if cached

    Returns:
        Parsed DialengConfig
    """
    global _config, _config_path

    if config_path is None:
        config_path = Path.cwd() / "dialeng_config.json"

    # Return cached if available and path matches
    if _config is not None and not force_reload and _config_path == config_path:
        return _config

    _config_path = config_path

    # Create default if doesn't exist
    if not config_path.exists():
        raw = _create_default_config(config_path)
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            logger.info(f"Loaded dialeng_config.json from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dialeng_config.json: {e}")
            print(f"   Warning: Invalid dialeng_config.json, using defaults")
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


def print_config_status(config: DialengConfig, detected_backend: Optional[str] = None) -> None:
    """Print config status for startup logging.

    Args:
        config: The parsed DialengConfig
        detected_backend: The detected backend from credential detection (for showing active default)
    """
    models = ", ".join(m.name for m in config.available_models)
    sdk_mode = "SDK direct" if config.use_sdk_directly else "claudette-agent"
    print(f"   Config: dialeng_config.json")
    print(f"      AWS Region:     {config.aws_region}")
    print(f"      Models:         {models}")

    # Show default models per provider
    bedrock_default = config.get_default_model("bedrock")
    claude_code_default = config.get_default_model("claude_code_subscription")

    # Highlight the active default based on detected backend
    if detected_backend == "bedrock":
        print(f"      Default Model:  {bedrock_default} (Bedrock) ← active")
        print(f"                      {claude_code_default} (Claude Code)")
    elif detected_backend == "claude_code_subscription":
        print(f"      Default Model:  {bedrock_default} (Bedrock)")
        print(f"                      {claude_code_default} (Claude Code) ← active")
    else:
        print(f"      Default Model:  {bedrock_default} (Bedrock)")
        print(f"                      {claude_code_default} (Claude Code)")

    print(f"      Default Mode:   {config.default_mode}")
    print(f"      LLM Provider:   {sdk_mode}")
    if config.debug_mode:
        print(f"      Debug Mode:     ON (logs to {config.debug_log_dir})")
    # Tool settings
    tools_status = "enabled" if config.tool_builtin_enabled else "disabled"
    confirm_status = "required" if config.tool_require_confirmation else "off"
    print(f"      Tool Calling:   {tools_status} (max {config.tool_max_steps} steps, confirm: {confirm_status})")
