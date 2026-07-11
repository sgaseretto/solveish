import json
from pathlib import Path

import pytest
from fasthtml.common import to_xml
from starlette.testclient import TestClient

import dialeng.app as appmod
from dialeng.services import dialeng_config
from dialeng.services.dialeng_config import DEFAULT_CONFIG, load_config, reset_config_cache
from dialeng.services.settings_service import (
    SettingDefinition,
    SettingsConflictError,
    SettingsValidationError,
    commit_settings,
    config_revision,
    register_dynamic_setting,
    settings_snapshot,
)
from dialeng.ui.settings import SettingsSidebar


def _write_config(path: Path, updates: dict | None = None) -> None:
    raw = json.loads(json.dumps(DEFAULT_CONFIG))
    if updates:
        raw.update(updates)
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_partial_commit_preserves_unrelated_and_custom_configuration(tmp_path):
    path = tmp_path / "dialeng_config.json"
    _write_config(path, {"extension_data": {"kept": True}})
    before = json.loads(path.read_text(encoding="utf-8"))

    config, snapshot, changed = commit_settings(
        path,
        {"display.font_size": 18},
        config_revision(before),
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert changed == ["display.font_size"]
    assert saved["display"]["font_size"] == 18
    assert saved["extension_data"] == {"kept": True}
    assert config.display_font_size == 18
    assert snapshot["revision"] == config_revision(saved)


def test_stale_revision_is_rejected_without_changing_file(tmp_path):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    original = path.read_text(encoding="utf-8")

    with pytest.raises(SettingsConflictError):
        commit_settings(path, {"display.font_size": 16}, "stale")

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        ({"display.font_size": 40}, "display.font_size"),
        ({"tool_settings.max_steps": 0}, "tool_settings.max_steps"),
        ({"colab.default_runtime": "quantum"}, "colab.default_runtime"),
        ({"unknown.path": True}, "unknown.path"),
    ],
)
def test_invalid_settings_return_field_errors(tmp_path, changes, path):
    config_path = tmp_path / "dialeng_config.json"
    _write_config(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    with pytest.raises(SettingsValidationError) as error:
        commit_settings(config_path, changes, config_revision(raw))

    assert path in error.value.errors


def test_snapshot_reports_effective_environment_source_and_prevents_edit(tmp_path, monkeypatch):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    monkeypatch.setenv("DIALENG_COLAB_AUTH_PROVIDER", "adc")

    snapshot = settings_snapshot(path)
    field = snapshot["fields"]["colab.auth_provider"]
    assert field["configured_value"] == "oauth_local"
    assert field["effective_value"] == "adc"
    assert field["source"] == "environment"
    assert field["managed"] is True

    with pytest.raises(SettingsValidationError):
        commit_settings(
            path,
            {"colab.auth_provider": "oauth_remote"},
            snapshot["revision"],
        )


def test_invalid_environment_override_reports_diagnostic_without_breaking_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    monkeypatch.setenv("DIALENG_COLAB_AUTH_PROVIDER", "invalid-provider")

    field = settings_snapshot(path)["fields"]["colab.auth_provider"]

    assert field["managed"] is True
    assert field["effective_value"] == "oauth_local"
    assert field["diagnostic"]


def test_extension_control_definition_can_participate_in_validated_updates(tmp_path):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    register_dynamic_setting(SettingDefinition("sample_extension.enabled", "bool", False))
    snapshot = settings_snapshot(path)

    _, _, changed = commit_settings(
        path,
        {"sample_extension.enabled": True},
        snapshot["revision"],
    )

    assert changed == ["sample_extension.enabled"]
    assert json.loads(path.read_text(encoding="utf-8"))["sample_extension"]["enabled"] is True


def test_config_save_failure_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    original = path.read_text(encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr(dialeng_config.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        dialeng_config.save_config({"display": {"font_size": 19}}, path)

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".dialeng_config.json.*.tmp"))


def test_settings_sidebar_exposes_persistence_and_authoritative_controls(tmp_path):
    path = tmp_path / "dialeng_config.json"
    _write_config(path)
    config = load_config(path, force_reload=True)
    html = to_xml(SettingsSidebar(config))

    assert "persistence.autosave.enabled" in html
    assert "persistence.autosave.recovery_snapshots" in html
    assert 'id="settings-search"' in html
    assert 'onsubmit="saveSettings(event)"' in html
    assert 'data-apply-mode="server_restart"' in html


def test_settings_routes_return_conflict_and_validation_statuses(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    previous_root = appmod.NOTEBOOKS_DIR

    async def no_activation(_revision, _changed_paths):
        return None

    monkeypatch.setattr(appmod, "_activate_settings", no_activation)
    try:
        appmod.set_root_dir(root)
        client = TestClient(appmod.app)
        snapshot = client.get("/settings").json()

        invalid = client.post("/settings", json={
            "revision": snapshot["revision"],
            "changes": {"display.font_size": 99},
        })
        assert invalid.status_code == 422
        assert "display.font_size" in invalid.json()["errors"]

        saved = client.post("/settings", json={
            "revision": snapshot["revision"],
            "changes": {"display.font_size": 17},
        })
        assert saved.status_code == 202
        assert saved.json()["changed_paths"] == ["display.font_size"]

        stale = client.post("/settings", json={
            "revision": snapshot["revision"],
            "changes": {"display.font_size": 18},
        })
        assert stale.status_code == 409
        assert stale.json()["error"] == "settings_conflict"
    finally:
        appmod.set_root_dir(previous_root)
        reset_config_cache()
