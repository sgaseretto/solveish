"""Keep extension authoring documentation aligned with the public API contract."""
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_user_extension_guide_covers_the_supported_end_to_end_workflow():
    guide = (ROOT / "docs" / "guides" / "extensions.md").read_text(encoding="utf-8")

    required = (
        "## End-To-End Authoring Workflow",
        "## Command Palette",
        "args_schema",
        "## Lock And Restore An Extension Environment",
        "extension sync --offline",
        "## Notebook Development Profile",
        "## Isolated And Privileged Modes",
        "## Troubleshooting",
        "## Release Checklist",
        "ExtensionSetting",
        "clean `uv` environment",
    )
    for phrase in required:
        assert phrase in guide
    assert "from dialeng.services.settings_service import SettingDefinition" not in guide


def test_llm_extension_skill_covers_runtime_trust_and_reproducibility():
    skill = (
        ROOT / "dialeng" / "builtin_skills" / "extension-creator" / "SKILL.md"
    ).read_text(encoding="utf-8")

    required = (
        "## Tool Map",
        "## Template Selection",
        "## Authoring Rules",
        "## Reproducibility And Packaging",
        "## Diagnosis And Recovery",
        "## Completion Checklist",
        "package_extension",
        "request_extension_activation",
        "python -I",
        "privileged",
        "sync never grants trust",
        "Never silently trust, activate, sync, install, or elevate",
    )
    for phrase in required:
        assert phrase in skill
