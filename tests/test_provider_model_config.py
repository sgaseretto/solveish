import pytest

from dialeng.services.dialeng_config import DEFAULT_CONFIG, ConfigValidationError, _parse_config


def test_provider_model_catalogs_are_separate_and_defaults_are_qualified():
    config = _parse_config(DEFAULT_CONFIG)
    assert [model.id for model in config.get_models_for_provider("openai_codex")] == [
        "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex",
    ]
    assert all(model.id.startswith("claude-") for model in config.get_models_for_provider("claudette:bedrock"))
    assert config.validate_model("openai_codex", "claude-sonnet-4-5") == "gpt-5.4"


def test_model_entries_require_a_supported_provider():
    with pytest.raises(ConfigValidationError, match="providers is required"):
        _parse_config({
            "models": {
                "available": [{"id": "custom-claude", "name": "Custom Claude"}],
                "defaults": {"bedrock": "custom-claude"},
            }
        })


def test_removed_provider_keys_are_rejected_with_actionable_message():
    with pytest.raises(ConfigValidationError, match="was removed"):
        _parse_config({"llm": {"claude" + "_agent_sdk": True}})
