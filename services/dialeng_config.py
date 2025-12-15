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
            {"id": "claude-sonnet-3-7", "name": "Claude Sonnet 3.7", "default": True},
            {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "default": False},
            {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "default": False}
        ],
        "anthropic_api_map": {
            "claude-sonnet-4-5": "claude-sonnet-4-5-20250514",
            "claude-haiku-4-5": "claude-haiku-4-5-20250514",
            "claude-sonnet-3-7": "claude-3-7-sonnet-20250219",
            "comment": "Model IDs for direct Anthropic API (with date suffix)"
        },
        "bedrock_map": {
            "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
            "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20250514-v1:0",
            "claude-sonnet-3-7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "comment": "Model IDs for AWS Bedrock (with region prefix and version suffix)"
        },
        "claudette_agent_map": {
            "claude-sonnet-4-5": "claude-sonnet-4-5",
            "claude-haiku-4-5": "claude-haiku-4-5",
            "claude-sonnet-3-7": "claude-sonnet-3-7",
            "comment": "Model IDs for claudette-agent (Claude Code subscription) - uses simple names"
        }
    },
    "modes": {
        "default": "mock",
        "comment": "Default dialog mode when opening a notebook. Options: mock, learning, concise, standard"
    }
}


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    name: str
    default: bool = False


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

    # Default mode
    default_mode: str = "mock"

    # Raw config for reference
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def get_default_model(self) -> str:
        """Get the default model ID."""
        for model in self.available_models:
            if model.default:
                return model.id
        return self.available_models[0].id if self.available_models else "claude-sonnet-4-5"

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
            name=m.get("name", m.get("id", "")),
            default=m.get("default", False)
        )
        for m in available
        if m.get("id")  # Skip entries without ID
    ]

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


def print_config_status(config: DialengConfig) -> None:
    """Print config status for startup logging."""
    models = ", ".join(m.name for m in config.available_models)
    default_model = config.get_default_model()
    print(f"   Config: dialeng_config.json")
    print(f"      AWS Region:     {config.aws_region}")
    print(f"      Models:         {models}")
    print(f"      Default Model:  {default_model}")
    print(f"      Default Mode:   {config.default_mode}")
