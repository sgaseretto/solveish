from dialeng.services.dialeng_config import _parse_config


def test_provider_model_catalogs_are_separate_and_defaults_are_qualified():
    config = _parse_config({})
    assert [model.id for model in config.get_models_for_provider("openai_codex")] == [
        "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex",
    ]
    assert all(model.id.startswith("claude-") for model in config.get_models_for_provider("claude_agent_sdk"))
    assert config.validate_model("openai_codex", "claude-sonnet-4-5") == "gpt-5.4"


def test_legacy_model_entries_are_inferred_as_claude_models():
    config = _parse_config({
        "models": {
            "available": [{"id": "custom-claude", "name": "Custom Claude"}],
            "defaults": {"anthropic_api": "custom-claude"},
        }
    })
    assert [m.id for m in config.get_models_for_provider("claudette:anthropic_api")] == ["custom-claude"]
    assert "custom-claude" not in [m.id for m in config.get_models_for_provider("openai_codex")]
