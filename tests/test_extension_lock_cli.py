"""CLI integration coverage for extension environment lock workflows."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from dialeng.services.extension_project_service import ExtensionProjectService


def _run(project: Path, home: Path, *arguments: str, check: bool = True):
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-c", "from dialeng.cli import cli; cli()", *arguments],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
        timeout=30,
    )


def test_cli_lock_check_update_and_sync_round_trip(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    created = ExtensionProjectService(project).create("cli-locked")
    extension_id = created["extension_id"]

    _run(project, home, "extension", "enable", extension_id, "--path", str(project))
    locked = _run(project, home, "extension", "lock", "--path", str(project))
    payload = json.loads(locked.stdout)
    assert payload["ok"] is True
    assert (project / ".dialeng" / "extensions.lock").is_file()

    checked = _run(project, home, "extension", "check", "--path", str(project))
    assert json.loads(checked.stdout)["ok"] is True

    module = Path(created["module"])
    module.write_text(module.read_text(encoding="utf-8") + "\nCHANGED = True\n", encoding="utf-8")
    drift = _run(
        project, home, "extension", "check", "--path", str(project), check=False,
    )
    assert drift.returncode == 1
    assert any(item["field"] == "source_hash" for item in json.loads(drift.stdout)["drift"])

    updated = _run(
        project, home, "extension", "update", extension_id, "--path", str(project),
    )
    assert any(item["requires_trust_review"] for item in json.loads(updated.stdout)["changes"])

    _run(project, home, "extension", "disable", extension_id, "--path", str(project))
    synced = _run(project, home, "extension", "sync", "--offline", "--path", str(project))
    synced_payload = json.loads(synced.stdout)
    assert synced_payload["ok"] is True
    assert synced_payload["enabled"] == [extension_id]
    assert synced_payload["trust_granted"] is False
    assert synced_payload["offline"] is True
